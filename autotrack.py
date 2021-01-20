"""
Python re-implemented of "Learning Spatial-Temporal Regularized Correlation Filters for Visual Tracking"
@inproceedings{li2018learning,
  title={Learning spatial-temporal regularized correlation filters for visual tracking},
  author={Li, Feng and Tian, Cheng and Zuo, Wangmeng and Zhang, Lei and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4904--4913},
  year={2018}
}
"""
import numpy as np
import cv2
# import time
from bacflibs.utils import cos_window,gaussian2d_rolled_labels
from bacflibs.fft_tools import fft2,ifft2
from numpy.fft import fft, ifft
# from bacflibs.feature import extract_hog_feature,extract_pyhog_feature,extract_cn_feature
from bacflibs.feature import extract_hog_feature,extract_cn_feature
import autotrack_config
from bacflibs.cf_utils import resp_newton,mex_resize,resize_dft2,circShift
# from bacflibs.scale_estimator import DSSTScaleEstimator

class AutoTrack(object):
    def __init__(self,config=autotrack_config.AutoTrackConfig()):
        super(AutoTrack).__init__()
        #sample and feature parameter
        self.name='AutoTrack'
        self.hog_cell_size = config.hog_cell_size
        self.hog_n_dim = config.hog_n_dim

        self.gray_cell_size = config.gray_cell_size
        self.cn_use_for_gray = config.cn_use_for_gray
        self.cn_cell_size = config.cn_cell_size
        self.cn_n_dim = config.cn_n_dim

        self.cell_size=self.hog_cell_size

        self.search_area_shape = config.search_area_shape
        self.search_area_scale=config.search_area_scale
        self.min_image_sample_size=config.min_image_sample_size
        self.max_image_sample_size=config.max_image_sample_size
        self.feature_downsample_ratio=config.feature_downsample_ratio
        self.reg_window_max=config.reg_window_max
        self.reg_window_min=config.reg_window_min
        
        # detection parameters
        self.refinement_iterations=config.refinement_iterations
        self.newton_iterations=config.newton_iterations
        self.clamp_position=config.clamp_position

        # learning parameters
        self.output_sigma_factor=config.output_sigma_factor
        self.nu=config.nu
        self.zeta=config.zeta
        self.delta=config.delta
        self.epsilon=config.epsilon
        self.lam=config.admm_lambda

        # ADMM params
        self.admm_max_iterations=config.max_iterations
        self.init_penalty_factor=config.init_penalty_factor
        self.max_penalty_factor=config.max_penalty_factor
        self.penalty_scale_step=config.penalty_scale_step

        # scale parameters
        self.number_of_scales =config.number_of_scales
        self.scale_step=config.scale_step

        self.use_mex_resize=True

        self.scale_type=config.scale_type
        self.scale_config = config.scale_config

        self.normalize_power=config.normalize_power
        self.normalize_size=config.normalize_size
        self.normalize_dim=config.normalize_dim
        self.square_root_normalization=config.square_root_normalization
        self.config=config


    def init(self,first_frame,bbox):

        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        self._center = (int(x0 + w / 2),int(y0 + h / 2))
        self.target_sz=(w,h)

        search_area=self.target_sz[0]*self.search_area_scale*self.target_sz[1]*self.search_area_scale
        self.sc=np.clip(1,a_min=np.sqrt(search_area/self.max_image_sample_size),a_max=np.sqrt(search_area/self.min_image_sample_size))

        self.base_target_sz=(self.target_sz[0]/self.sc,self.target_sz[1]/self.sc)

        if self.search_area_shape=='proportional':
            self.crop_size=(int(self.base_target_sz[0]*self.search_area_scale),int(self.base_target_sz[1]*self.search_area_scale))
        elif self.search_area_shape=='square':
            w=int(np.sqrt(self.base_target_sz[0]*self.base_target_sz[1])*self.search_area_scale)
            self.crop_size=(w,w)
        elif self.search_area_shape=='fix_padding':
            tmp=int(np.sqrt(self.base_target_sz[0]*self.search_area_scale+(self.base_target_sz[1]-self.base_target_sz[0])/4))+\
                (self.base_target_sz[0]+self.base_target_sz[1])/2
            self.crop_size=(self.base_target_sz[0]+tmp,self.base_target_sz[1]+tmp)
        else:
            raise ValueError
        output_sigma = np.sqrt(np.floor(self.base_target_sz[0]/self.cell_size)*np.floor(self.base_target_sz[1]/self.cell_size))*\
            self.output_sigma_factor

        self.crop_size = (int(round(self.crop_size[0] / self.cell_size) * self.cell_size),
                          int(round(self.crop_size[1] / self.cell_size) * self.cell_size))
        self.feature_map_sz = (self.crop_size[0] // self.cell_size, self.crop_size[1] // self.cell_size)
        y=gaussian2d_rolled_labels(self.feature_map_sz,output_sigma)

        self.cosine_window=(cos_window((y.shape[1],y.shape[0])))
#        self.cosine_window=self.cosine_window[1:-1,1:-1]
        self.yf=fft2(y)
        self.reg_scale=(int(np.floor(self.base_target_sz[0]/self.feature_downsample_ratio)),
                   int(np.floor(self.base_target_sz[1] / self.feature_downsample_ratio)))
        use_sz = self.feature_map_sz
        self.interp_sz=use_sz

        self.range_h,self.range_w,self.reg_window=self.create_reg_window_const(self.reg_scale,use_sz,self.reg_window_max,self.reg_window_min)

        self.ky = np.roll(np.arange(-int(np.floor((self.feature_map_sz[1] - 1) / 2)),
                                    int(np.ceil((self.feature_map_sz[1] - 1) / 2 + 1))),
                          -int(np.floor((self.feature_map_sz[1] - 1) / 2)))
        self.kx = np.roll(np.arange(-int(np.floor((self.feature_map_sz[0] - 1) / 2)),
                                    int(np.ceil((self.feature_map_sz[0] - 1) / 2 + 1))),
                          -int(np.floor((self.feature_map_sz[0] - 1) / 2)))

        if self.number_of_scales>0:
            self._min_scale_factor = self.scale_step ** np.ceil(
                np.log(np.max(5 / np.array(([self.crop_size[0], self.crop_size[1]])))) / np.log(self.scale_step))
            self._max_scale_factor = self.scale_step ** np.floor(np.log(np.min(
                first_frame.shape[:2] / np.array([self.base_target_sz[1], self.base_target_sz[0]]))) / np.log(
                self.scale_step))
            #print(self._min_scale_factor)
            #print(self._max_scale_factor)

        self.scale_estimator = DSSTScaleEstimator(self.target_sz, config=self.scale_config)
        self.scale_estimator.init(first_frame, self._center, self.base_target_sz, self.sc)
#            self._num_scales = self.scale_estimator.num_scales
#            self._scale_step = self.scale_estimator.scale_step
#
#            self._min_scale_factor = self._scale_step ** np.ceil(
#                np.log(np.max(5 / np.array(([self.crop_size[0], self.crop_size[1]])))) / np.log(self._scale_step))
#            self._max_scale_factor = self._scale_step ** np.floor(np.log(np.min(
#                first_frame.shape[:2] / np.array([self.base_target_sz[1], self.base_target_sz[0]]))) / np.log(
#                self._scale_step))
#        elif self.scale_type=='LP':
#            self.scale_estimator=LPScaleEstimator(self.target_sz,config=self.scale_config)
#            self.scale_estimator.init(first_frame,self._center,self.base_target_sz,self.sc)


        patch = self.get_sub_window(first_frame, self._center, model_sz=self.crop_size,
                                    scaled_sz=(int(np.round(self.crop_size[0] * self.sc)),
                                               int(np.round(self.crop_size[1] * self.sc))))
        patch=patch[:,:,[2,1,0]]
        xl_hc = self.extrac_hc_feature(patch, self.cell_size)
#        xl_hc =self.extrac_feature_test(patch, (50,50))
        xlf_hc = fft2(xl_hc * self.cosine_window[:, :, None])
        mu=0
        self.occ=False
        self.frame_id=1
        self.g_pre=np.zeros_like(xlf_hc)
        if self.occ==False:
            self.ADMM(xlf_hc,mu)


    def update(self,current_frame,frame_id,vis=False):
        self.frame_id=frame_id
        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3
#        old_pos=(np.inf,np.inf)
#        iter=1
#        while iter<=self.refinement_iterations and (np.abs(old_pos[0]-self._center[0])>1e-2 or
#                                                    np.abs(old_pos[1]-self._center[1])>1e-2):
#
#            sample_scales=self.sc*self.scale_factors
        
        #position search
        sample_pos=(int(np.round(self._center[0])),int(np.round(self._center[1])))
        sub_window = self.get_sub_window(current_frame, sample_pos, model_sz=self.crop_size,
                                                 scaled_sz=(int(round(self.crop_size[0] * self.sc)),
                                                            int(round(self.crop_size[1] * self.sc))))
        sub_window=sub_window[:,:,[2,1,0]]
#        xt_hc =self.extrac_feature_test(sub_window, (50,50))
        xt_hc = self.extrac_hc_feature(sub_window, self.cell_size)
#            for scale in sample_scales:
#                sub_window = self.get_sub_window(current_frame, sample_pos, model_sz=self.crop_size,
#                                                 scaled_sz=(int(round(self.crop_size[0] * scale)),
#                                                            int(round(self.crop_size[1] * scale))))
#                hc_features=self.extrac_hc_feature(sub_window, self.cell_size)[:,:,:,np.newaxis]
#                if xt_hc is None:
#                    xt_hc = hc_features
#                else:
#                    xt_hc = np.concatenate((xt_hc, hc_features), axis=3)
        xtw_hc=xt_hc*self.cosine_window[:,:,None]
        xtf_hc=fft2(xtw_hc)
        responsef_hc=np.sum(np.conj(self.g_f)[:,:,:]*xtf_hc,axis=2)
        responsef_padded=resize_dft2(responsef_hc,self.interp_sz)
        response = np.real(ifft2(responsef_padded))
        disp_row,disp_col,sind=resp_newton(response,responsef_padded,self.newton_iterations,self.ky,self.kx,self.feature_map_sz)
        if frame_id>2:
            response_shift=circShift(response,[-int(np.floor(disp_row)),-int(np.floor(disp_col))])
            response_pre_shift=circShift(self.response_pre,[-int(np.floor(self.disp_row_pre)),-int(np.floor(self.disp_col_pre))])
            response_diff=np.abs(np.abs(response_shift-response_pre_shift)/response_pre_shift)
            self.ref_mu, self.occ = self.updateRefmu(response_diff,self.zeta,self.nu,frame_id)
            response_diff=circShift(response_diff,[int(np.floor(response_diff.shape[0]/2)),int(np.floor(response_diff.shape[1]/2))])
            varience=self.delta*np.log(response_diff[self.range_h[0]:(self.range_h[-1]+1), self.range_w[0]:(self.range_w[-1]+1)]+1)
            self.reg_window[self.range_h[0]:(self.range_h[-1]+1), self.range_w[0]:(self.range_w[-1]+1)] = varience
        self.response_pre=response
        self.disp_row_pre=disp_row
        self.disp_col_pre=disp_col
            #row, col, sind = np.unravel_index(np.argmax(response, axis=None), response.shape)

            #disp_row = (row+ int(np.floor(self.feature_map_sz[1] - 1) / 2)) % self.feature_map_sz[1] - int(
            #    np.floor((self.feature_map_sz[1] - 1) / 2))
            #disp_col = (col + int(np.floor(self.feature_map_sz[0] - 1) / 2)) % self.feature_map_sz[0] - int(
            #    np.floor((self.feature_map_sz[0] - 1) / 2))

#            if vis is True:
#                self.score = response[:, :, sind].astype(np.float32)
#                self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
#                self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)
        dx, dy = (disp_col * self.cell_size*self.sc), (disp_row * self.cell_size*self.sc)
        trans_vec=np.array(np.round([dx,dy]))
#        old_pos = self._center
        self._center = (np.round(sample_pos[0] +dx), np.round(sample_pos[1] + dy))
#        print(self._center)
#        self.sc=self.sc*scale_change_factor
#        self.sc = np.clip(self.sc, self._min_scale_factor, self._max_scale_factor)
        
        #scale search and update
        self.sc = self.scale_estimator.update(current_frame, self._center, self.base_target_sz,
                                              self.sc)
#        print(self.sc)
        if self.scale_type == 'normal':
            self.sc = np.clip(self.sc, a_min=self._min_scale_factor,
                                                a_max=self._max_scale_factor)
        
        #training
        shift_sample_pos=np.array(2.0*np.pi*trans_vec/(self.sc*np.array(self.crop_size,dtype=float)))
#        shift_sample() hard to ensure objectiveness....
#        consider using original ways-----LBW
        # xlf_hc=self.shift_sample(xtf_hc, shift_sample_pos, self.kx, self.ky.T)
        patch = self.get_sub_window(current_frame, self._center, model_sz=self.crop_size,
                                          scaled_sz=(int(np.round(self.crop_size[0] * self.sc)),
                                                    int(np.round(self.crop_size[1] * self.sc))))
        patch=patch[:,:,[2,1,0]]
        xl_hc =self.extrac_feature_test(patch, (50,50))
        xl_hc = self.extrac_hc_feature(patch, self.cell_size)
        xlw_hc = xl_hc * self.cosine_window[:, :, None]
        xlf_hc = fft2(xlw_hc)
        mu = self.zeta
        if self.occ==False:
            self.ADMM(xlf_hc,mu)
        target_sz=(self.base_target_sz[0]*self.sc,self.base_target_sz[1]*self.sc)
        return [(self._center[0] - (target_sz[0]) / 2), (self._center[1] -(target_sz[1]) / 2), target_sz[0],target_sz[1]]

    def extrac_hc_feature(self,patch,cell_size,normalization=False):
        hog_features=extract_hog_feature(patch,cell_size)
        cn_features=extract_cn_feature(patch,cell_size)
        hc_features=np.concatenate((hog_features,cn_features),axis=2)
#        hc_features=hog_features
        if normalization is True:
            hc_features=self._feature_normalization(hc_features)
        return hc_features

    def get_sub_window(self, img, center, model_sz, scaled_sz=None):
        model_sz = (int(model_sz[0]), int(model_sz[1]))
        if scaled_sz is None:
            sz = model_sz
        else:
            sz = scaled_sz
        sz = (max(int(sz[0]), 2), max(int(sz[1]), 2))

        """
        w,h=sz
        xs = (np.floor(center[0]) + np.arange(w) - np.floor(w / 2)).astype(np.int64)
        ys = (np.floor(center[1]) + np.arange(h) - np.floor(h / 2)).astype(np.int64)
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= img.shape[1]] = img.shape[1] - 1
        ys[ys >= img.shape[0]] = img.shape[0] - 1
        im_patch = img[ys, :][:, xs]
        """
        im_patch = cv2.getRectSubPix(img, sz, center)
        if scaled_sz is not None:
            im_patch = mex_resize(im_patch, model_sz)
        return im_patch.astype(np.uint8)

    def ADMM(self,xlf,mu):
        model_xf = xlf
        self.l_f = np.zeros_like(model_xf)
        self.g_f = np.zeros_like(self.l_f)
        self.h_f = np.zeros_like(self.l_f)
        self.gamma =self.init_penalty_factor
        self.gamma_max =self.max_penalty_factor
        self.gamma_scale_step = self.penalty_scale_step
        T = self.feature_map_sz[0] * self.feature_map_sz[1]
        S_xx = np.sum(np.conj(model_xf) * model_xf, axis=2)
        Sg_pre_f = np.sum(np.conj(model_xf) * self.g_pre, axis=2)
        Sgx_pre_f = model_xf * Sg_pre_f[:, :, None]
        iter = 1
        while iter <= self.admm_max_iterations:
            #subproblem g
            B = S_xx + T * (self.gamma + mu)
            Slx_f = np.sum(np.conj(model_xf) * self.l_f, axis=2)
            Shx_f = np.sum(np.conj(model_xf) * self.h_f, axis=2)

            tmp0 = (1 / (T * (self.gamma + mu)) * (self.yf[:, :, None] * model_xf)) - ((1 / (self.gamma + mu)) * self.l_f) + (
                    self.gamma / (self.gamma + mu)) * self.h_f + \
                   (mu / (self.gamma + mu)) * self.g_pre
            tmp1 = 1 / (T * (self.gamma + mu)) * (model_xf * ((S_xx * self.yf)[:, :, None]))
            tmp2 = mu / (self.gamma + mu) * Sgx_pre_f
            tmp3 = 1 / (self.gamma + mu) * (model_xf * (Slx_f[:, :, None]))
            tmp4 = self.gamma / (self.gamma + mu) * (model_xf * Shx_f[:, :, None])
            self.g_f = (tmp0 - (tmp1 + tmp2 - tmp3 +tmp4) / B[:, :, None]).astype(np.complex64)
            #subproblem h
            self.h_f = fft2(self.argmin_g(self.reg_window, self.gamma, T, ifft2(self.gamma * (self.g_f + self.l_f))))
            #subproblem mu
            if self.frame_id>2 and iter<self.admm_max_iterations:
                for i in range(self.g_f.shape[2]):
                    z=np.power(np.linalg.norm((self.g_f[:,:,i]-self.g_pre[:,:,i]),2),2)/(2*self.epsilon)
                    mu=self.ref_mu-z
            # update l
            self.l_f = self.l_f + (self.gamma * (self.g_f - self.h_f))
            # update gama
            self.gamma = min(self.gamma_scale_step * self.gamma, self.gamma_max)
            iter += 1
        self.g_pre=self.g_f


    def argmin_g(self,w0,zeta,T,X):
        lhd = T / (self.lam* w0 ** 2 + T*zeta)
        m = lhd[:, :, None] * X
        return m

#    def create_reg_window(self,reg_scale,use_sz,p,reg_window_max,reg_window_min,alpha,beta):
#        range_ = np.zeros((2, 2))
#        for j in range(len(use_sz)):
#            if use_sz[0]%2==1 and use_sz[1]%2==1:
#                if int(reg_scale[j]) % 2 == 1:
#                    range_[j, :] = np.array([-np.floor(use_sz[j] / 2), np.floor(use_sz[j] / 2)])
#                else:
#                    range_[j, :] = np.array([-(use_sz[j] / 2 - 1), (use_sz[j] / 2)])
#            else:
#                if int(reg_scale[j]) % 2 == 1:
#                    range_[j, :] = np.array([-np.floor(use_sz[j] / 2), (np.floor((use_sz[j] - 1) / 2))])
#                else:
#                    range_[j, :] = np.array([-((use_sz[j] - 1) / 2),((use_sz[j] - 1) / 2)])
#        wrs = np.arange(range_[1, 0], range_[1, 1] + 1)
#        wcs = np.arange(range_[0, 0], range_[0, 1] + 1)
#        wrs, wcs = np.meshgrid(wrs, wcs)
#        res = (np.abs(wrs) / reg_scale[1]) ** p + (np.abs(wcs) / reg_scale[0]) ** p
#        reg_window = reg_window_max / (1 + np.exp(-1. * alpha * (np.power(res, 1. / p) -beta))) +reg_window_min
#        reg_window=reg_window.T
#        return reg_window

    def create_reg_window_const(self, reg_scale, use_sz,reg_window_max, reg_window_min):
        reg_window=np.ones((use_sz[1],use_sz[0]))*reg_window_max
        range_=np.zeros((2,2))
        for j in range(2):
            range_[j,:]=np.array([0,reg_scale[j]-1])-np.floor(reg_scale[j]/2)
        cx=int(np.floor((use_sz[0]+1)/2))+(use_sz[0]+1)%2-1
        cy=int(np.floor((use_sz[1]+1)/2))+(use_sz[1]+1)%2-1
        range_h=np.arange(cy+range_[1,0],cy+range_[1,1]+1).astype(np.int)
        range_w=np.arange(cx+range_[0,0],cx+range_[0,1]+1).astype(np.int)
        a_h,a_w=np.meshgrid(range_h,range_w)
        reg_window[a_h,a_w]=reg_window_min
        return range_h,range_w,reg_window

    def _feature_normalization(self, x):
        if hasattr(self.config, 'normalize_power') and self.config.normalize_power > 0:
            if self.config.normalize_power == 2:
                x = x * np.sqrt((x.shape[0]*x.shape[1]) ** self.config.normalize_size * (x.shape[2] ** self.config.normalize_dim) / (x ** 2).sum(axis=(0, 1, 2)))
            else:
                x = x * ((x.shape[0]*x.shape[1]) ** self.config.normalize_size) * (x.shape[2] ** self.config.normalize_dim) / ((np.abs(x) ** (1. / self.config.normalize_power)).sum(axis=(0, 1, 2)))

        if self.config.square_root_normalization:
            x = np.sign(x) * np.sqrt(np.abs(x))
        return x.astype(np.float32)
    
    def shift_sample(self, xf, shift, kx, ky):
        xp = np
        shift_exp_y = [xp.exp(1j * shift[0] * ky_).astype(xp.complex64) for ky_ in ky]
        shift_exp_x = [xp.exp(1j * shift[1] * kx_).astype(xp.complex64) for kx_ in kx]
        xf = [xf_ * sy_.reshape(1, -1) * sx_.reshape((-1, 1))
                for xf_, sy_, sx_ in zip(xf, shift_exp_y, shift_exp_x)]
        return xf
    
    def updateRefmu(self,response_dif,init_mu,p,frame):
        phi=0.3
        m=init_mu
        eta=np.linalg.norm(response_dif,2)/10000
        if eta<phi:
            y=m/(1+np.log(p*eta+1))
            occ=False
        else:
            y=50
            occ=True
        return np.float16(y),occ
    
    def extrac_feature_test(self,patch,sz):
        total_dim=42
        im=np.sum(patch,axis=2)/300
        resized_patch=cv2.resize(im,sz,interpolation = cv2.INTER_AREA)
        w,h=resized_patch.shape
        feature_pixels=np.zeros([w,h,total_dim])
        for i in range(total_dim):
            feature_pixels[:,:,i]=resized_patch
        return feature_pixels
    
#    def init_regwindow(self,sz,target_sz):
#        reg_scale=target_sz
#        use_sz=sz
#        reg_window=np.ones(use_sz)*self.reg_window_max
#        ran=np.zeros(reg_scale.size,2)
#        
#        for j in range(reg_scale.size):
#            ran[j,:]=[0,reg_scale[j]-1]-np.floor(reg_scale[j]/2)
#        center=np.floor((use_sz)/2)+np.mod(use_sz+1,2)
#        range_h=range((center[0]+ran[0,0]),(center[0]+ran[0,1]))
#        range_w=range((center[1]+ran[1,0]),(center[1]+ran[1,1]))
#        reg_window[range_h,range_w]=self.reg_window_min
#        return range_h,range_w,reg_windowimport matplotlib.pyplot as plt
class DSSTScaleEstimator:
    def __init__(self,target_sz,config):
        init_target_sz = np.array([target_sz[0],target_sz[1]])

        self.config=config
        num_scales = self.config.number_of_scales_filter
        scale_step = self.config.scale_step_filter
        scale_sigma = np.sqrt(self.config.number_of_scales_filter) * self.config.scale_sigma_factor

        scale_exp = np.arange(-np.floor(num_scales - 1)/2,
                              np.ceil(num_scales-1)/2+1,
                              dtype=np.float32)

#        interp_scale_exp = np.arange(-np.floor((self.config.number_of_interp_scales - 1) / 2),
#                                     np.ceil((self.config.number_of_interp_scales - 1) / 2) + 1,
#                                     dtype=np.float32)

        self.scale_size_factors = scale_step ** (-scale_exp)
#        self.interp_scale_factors = scale_step ** interp_scale_exp_shift

        ys = np.exp(-0.5 * (scale_exp ** 2) / (scale_sigma ** 2))
        self.yf = fft(ys)
        self.window = np.hanning(ys.shape[0]).T.astype(np.float32)
        # make sure the scale model is not to large, to save computation time


        self.num_scales = num_scales
        self.scale_step = scale_step

        if self.config.scale_model_factor ** 2 * np.prod(init_target_sz) > self.config.scale_model_max_area:
            scale_model_factor = np.sqrt(self.config.scale_model_max_area / np.prod(init_target_sz))
        else:
            scale_model_factor = self.config.scale_model_factor

        # set the scale model size
        self.scale_model_sz = np.floor(init_target_sz * scale_model_factor)
#        self.max_scale_dim = self.config.s_num_compressed_dim == 'MAX'
#        if self.max_scale_dim:
#            self.s_num_compressed_dim = len(self.scale_size_factors)
#        else:
#            self.s_num_compressed_dim = self.config.s_num_compressed_dim



    def init(self,im,pos,base_target_sz,current_scale_factor):

        # self.scale_factors = np.array([1])
        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz,self.window)
        # compute projection basis
#        if self.max_scale_dim:
#            self.basis, _ = scipy.linalg.qr(self.s_num, mode='economic')
#            scale_basis_den, _ = scipy.linalg.qr(xs, mode='economic')
#        else:
#            U, _, _ = np.linalg.svd(self.s_num)
#            self.basis = U[:, :self.s_num_compressed_dim]
#            V, _, _ = np.linalg.svd(xs)
#            scale_basis_den = V[:, :self.s_num_compressed_dim]
#            self.basis = self.basis.T
        # compute numerator
#        feat_proj = self.basis.dot(self.s_num) * self.window
        xsf = np.fft.fft(xs, axis=1)
        self.sf_num = self.yf * np.conj(xsf)

        # update denominator
#        xs = scale_basis_den.T.dot(xs)*self.window
        new_sf_den = np.sum(xsf*np.conj(xsf), 0)
        self.sf_den = new_sf_den


    def update(self, im, pos, base_target_sz, current_scale_factor):
#        base_target_sz=np.array([base_target_sz[0],base_target_sz[1]])
        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz,self.window)

        # get scores
        xsf = np.fft.fft(xs, axis=1)
        scale_responsef = np.sum(self.sf_num * xsf, 0) / (self.sf_den + self.config.scale_lambda)
        recovered_scale_index = np.argmax(np.real(ifft(scale_responsef)))

        current_scale_factor=current_scale_factor*self.scale_size_factors[recovered_scale_index]

        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz,self.window)
        #(error)xs = self._shift_scale_sample(im, pos, base_target_sz, xs, recovered_scale_index, scales, self.window, self.scale_model_sz)
        xsf = np.fft.fft(xs, axis=1)
        new_sf_num = self.yf * np.conj(xsf)
        new_sf_den = np.sum(xsf*np.conj(xsf), 0)
        self.sf_num = (1 - self.config.scale_learning_rate) * self.sf_num + self.config.scale_learning_rate * new_sf_num
        self.sf_den = (1 - self.config.scale_learning_rate) * self.sf_den + self.config.scale_learning_rate * new_sf_den
        return current_scale_factor


    def _extract_scale_sample(self, im, pos, base_target_sz, scale_factors, scale_model_sz, window):
        scale_sample = []
        base_target_sz=np.array([base_target_sz[0],base_target_sz[1]])
        for idx, scale in enumerate(scale_factors):
            patch_sz = np.floor(base_target_sz * scale)
            im_patch=cv2.getRectSubPix(im,(int(patch_sz[0]),int(patch_sz[1])),pos)
            if scale_model_sz[0] > patch_sz[1]:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
            im_patch_resized = cv2.resize(im_patch, (int(scale_model_sz[0]),int(scale_model_sz[1])), interpolation=interpolation).astype(np.uint8)
            im_patch_resized=im_patch_resized[:,:,[2,1,0]]
            temp=extract_hog_feature(im_patch_resized,cell_size=4)
#            temp=self.extrac_feature_test(im_patch, (4,8), 31)
            scale_sample.append(temp.reshape((-1,1),order="F")*window[idx])
        scale_sample = np.concatenate(scale_sample, axis=1)
        return scale_sample
    
    def _shift_scale_sample(self,im, pos, base_target_sz, xs, recovered_scale, scaleFactors,scale_window, scale_model_sz):
        nScales=len(scaleFactors)
        base_target_sz=np.array([base_target_sz[0],base_target_sz[1]])
        out=[]
        shift_pos=int(recovered_scale-np.ceil(nScales/2))
        if shift_pos==0:
            out=xs
        elif shift_pos>0:
            for j in range(nScales-shift_pos):
                xin=np.expand_dims(xs[:,j+shift_pos],axis=1)
                out.append(xin/(scale_window[j+shift_pos]+0.00001)*scale_window[j])
            for i in range(shift_pos):
                patch_sz = np.floor(base_target_sz * scaleFactors[nScales-shift_pos+i])
                im_patch=cv2.getRectSubPix(im,(int(patch_sz[0]),int(patch_sz[1])),pos)
                if scale_model_sz[0] > patch_sz[1]:
                    interpolation = cv2.INTER_LINEAR
                else:
                    interpolation = cv2.INTER_AREA
                im_patch_resized = cv2.resize(im_patch, (int(scale_model_sz[0]),int(scale_model_sz[1])), interpolation=interpolation).astype(np.uint8)
                im_patch_resized=im_patch_resized[:,:,[2,1,0]]
                temp=extract_hog_feature(im_patch_resized,cell_size=4)
#                temp=self.extrac_feature_test(im_patch, (4,8), 31)
                out.append(temp.reshape((-1, 1),order="F")*scale_window[nScales-shift_pos+i])
            out=np.concatenate(out, axis=1)
        else:
            for i in range(-shift_pos):
                patch_sz = np.floor(base_target_sz * scaleFactors[i])
                im_patch=cv2.getRectSubPix(im,(int(patch_sz[0]),int(patch_sz[1])),pos)
                if scale_model_sz[0] > patch_sz[1]:
                    interpolation = cv2.INTER_LINEAR
                else:
                    interpolation = cv2.INTER_AREA
                im_patch_resized = cv2.resize(im_patch, (int(scale_model_sz[0]),int(scale_model_sz[1])), interpolation=interpolation).astype(np.uint8)
                im_patch_resized=im_patch_resized[:,:,[2,1,0]]
                temp=extract_hog_feature(im_patch_resized,cell_size=4)
#                temp=self.extrac_feature_test(im_patch, (8,4), 31)
                out.append(temp.reshape((-1, 1),order="F")*scale_window[i])
            for j in range(nScales+shift_pos):
                xin=np.expand_dims(xs[:,j],axis=1)
                out.append(xin/(scale_window[j]+0.00001)*scale_window[j-shift_pos])
            out=np.concatenate(out, axis=1)
        return out
    
    def extrac_feature_test(self,patch,sz,dim):
        total_dim=dim
        im=np.sum(patch,axis=2)/300
        resized_patch=cv2.resize(im,sz,interpolation = cv2.INTER_AREA)
        w,h=resized_patch.shape
        feature_pixels=np.zeros([w,h,total_dim])
        for i in range(total_dim):
            feature_pixels[:,:,i]=resized_patch
        return feature_pixels    







