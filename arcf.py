"""
Python re-implementation of "Learning Background-Aware Correlation Filters for Visual Tracking"
@article{Galoogahi2017Learning,
  title={Learning Background-Aware Correlation Filters for Visual Tracking},
  author={Galoogahi, Hamed Kiani and Fagg, Ashton and Lucey, Simon},
  year={2017},
}
"""
import numpy as np
import cv2
#import torch
#import torch.nn.functional as F
from bacflibs.utils import cos_window,gaussian2d_rolled_labels
#libs.dcf hann2d & label
from bacflibs.fft_tools import fft2,ifft2,ifft2_sym,cifft2
#libs.fourier 正负傅里叶
from bacflibs.feature import extract_hog_feature,extract_cn_feature
#maybe in 'features'
# from bacflibs.bacf_config import BACFConfig
#BACF configuration-parameters
from bacflibs.cf_utils import mex_resize,resp_newton,resize_dft2
#libs.optimization  need to figure out what 'resize' means
from bacflibs.scale_estimator import LPScaleEstimator
#No dependency here

from collections import namedtuple


class ARCF(object):
    def __init__(self, net_path=None, **kargs):
        self.name = 'ARCF'
        self.params = self.parse_args(**kargs)

        # setup GPU device if available
#        self.cuda = torch.cuda.is_available()
        self.device = 'cpu'

    def parse_args(self, **kargs):
        # default parameters
        params = {
            # inference parameters
            'cell_size': 4,
            'cell_selection_thresh': 0.75**2,
            'search_area_shape': 'square',
            'search_area_scale': 5,
            'filter_max_area': 50**2,
            'interp_factor': 0.0190,
            'output_sigma_factor': 1./16,
            'interpolate_response': 4,
            'newton_iterations': 50,
            'number_of_scales': 5,
            'scale_step': 1.01,
            'admm_iterations': 2,
            'admm_lambda': 0.01,
            'admm_gamma': 0.71}
        for key, val in kargs.items():
            if key in params:
                params.update({key: val})
        return namedtuple('GenericDict', params.keys())(**params)
    def init(self, image, box):

        self.cell_size=self.params.cell_size
        self.cell_selection_thresh=self.params.cell_selection_thresh
        self.search_area_shape = self.params.search_area_shape
        self.search_area_scale=self.params.search_area_scale
        self.filter_max_area = self.params.filter_max_area
        self.interp_factor=self.params.interp_factor
        self.output_sigma_factor = self.params.output_sigma_factor
        self.interpolate_response =self.params.interpolate_response
        self.newton_iterations =self.params.newton_iterations
        self.number_of_scales =self.params.number_of_scales
        self.scale_step = self.params.scale_step
        self.admm_iterations = self.params.admm_iterations
        self.admm_lambda = self.params.admm_lambda
        self.admm_gamma = self.params.admm_gamma
        self.frame = 1
        # self.learning_rate_scale = self.params.learning_rate_scale
        # self.scale_sz_window = self.params.scale_sz_window

        # class ScaleConfig:
            # learning_rate_scale = self.learning_rate_scale
            # scale_sz_window = self.scale_sz_window

        # self.scale_config = ScaleConfig()

        # Get target position and size
        # state = info['init_bbox']
        state = box
        bbox = np.array(state).astype(np.int64)
        x, y, w, h = tuple(bbox)
        self._center = (x + w / 2, y + h / 2)
        # self._center = tuple(np.floor(self._center))
        self.w, self.h = w, h
        self.feature_ratio=self.cell_size
        self.search_area=(self.w/self.feature_ratio*self.search_area_scale)*\
                         (self.h/self.feature_ratio*self.search_area_scale)
        if self.search_area<self.cell_selection_thresh*self.filter_max_area:
            self.cell_size=int(min(self.feature_ratio,max(1,int(np.ceil(np.sqrt(
                self.w*self.search_area_scale/(self.cell_selection_thresh*self.filter_max_area)*\
                self.h*self.search_area_scale))))))
            self.feature_ratio=self.cell_size
            self.search_area = (self.w / self.feature_ratio * self.search_area_scale) * \
                               (self.h / self.feature_ratio * self.search_area_scale)

        if self.search_area>self.filter_max_area:
            self.current_scale_factor=np.sqrt(self.search_area/self.filter_max_area)
        else:
            self.current_scale_factor=1.

        self.base_target_sz=(self.w/self.current_scale_factor,self.h/self.current_scale_factor)
        # self.target_sz=self.base_target_sz
        if self.search_area_shape=='proportional':
            self.crop_size=(int(self.base_target_sz[0]*self.search_area_scale),int(self.base_target_sz[1]*self.search_area_scale))
        elif self.search_area_shape=='square':
            w= np.sqrt(self.base_target_sz[0]*self.base_target_sz[1])*self.search_area_scale
            self.crop_size=(w,w)
        elif self.search_area_shape=='fix_padding':
            tmp=int(np.sqrt(self.base_target_sz[0]*self.search_area_scale+(self.base_target_sz[1]-self.base_target_sz[0])/4))+\
                (self.base_target_sz[0]+self.base_target_sz[1])/2
            self.crop_size=(self.base_target_sz[0]+tmp,self.base_target_sz[1]+tmp)
        else:
            raise ValueError
        self.crop_size=(int(round(self.crop_size[0]/self.feature_ratio)*self.feature_ratio),int(round(self.crop_size[1]/self.feature_ratio)*self.feature_ratio))
        self.feature_map_sz=(self.crop_size[0]//self.feature_ratio,self.crop_size[1]//self.feature_ratio)
        output_sigma=np.sqrt(np.floor(self.base_target_sz[0]/self.feature_ratio)*np.floor(self.base_target_sz[1]/self.feature_ratio))*self.output_sigma_factor
        y=gaussian2d_rolled_labels(self.feature_map_sz, output_sigma)
        self.yf=fft2(y)
        if self.interpolate_response==1:
            self.interp_sz=(self.feature_map_sz[0]*self.feature_ratio,self.feature_map_sz[1]*self.feature_ratio)
        else:
            self.interp_sz=(self.feature_map_sz[0],self.feature_map_sz[1])
        self._window=cos_window(self.feature_map_sz)
        if self.number_of_scales>0:
            scale_exp=np.arange(-int(np.floor((self.number_of_scales-1)/2)),int(np.ceil((self.number_of_scales-1)/2))+1)
            self.scale_factors=self.scale_step**scale_exp
            self.min_scale_factor=self.scale_step**(np.ceil(np.log(max(5/self.crop_size[0],5/self.crop_size[1]))/np.log(self.scale_step)))
            self.max_scale_factor=self.scale_step**(np.floor(np.log(min(image.shape[0]/self.base_target_sz[1],
                                                                        image.shape[1]/self.base_target_sz[0]))/np.log(self.scale_step)))
        if self.interpolate_response>=3:
            self.ky=np.roll(np.arange(-int(np.floor((self.feature_map_sz[1]-1)/2)),int(np.ceil((self.feature_map_sz[1]-1)/2+1))),
                            -int(np.floor((self.feature_map_sz[1]-1)/2)))
            self.kx=np.roll(np.arange(-int(np.floor((self.feature_map_sz[0]-1)/2)),int(np.ceil((self.feature_map_sz[0]-1)/2+1))),
                            -int(np.floor((self.feature_map_sz[0]-1)/2))).T
            # self.kx = self.kx.reshape(self.kx.shape[0],1)
            # self.kx=np.roll(np.arange(-int(np.floor((self.feature_map_sz[1]-1)/2)),int(np.ceil((self.feature_map_sz[1]-1)/2+1))),
            #                 -int(np.floor((self.feature_map_sz[1]-1)/2)),axis=0)
            # self.ky=np.roll(np.arange(-int(np.floor((self.feature_map_sz[0]-1)/2)),int(np.ceil((self.feature_map_sz[0]-1)/2+1))),
            #                 -int(np.floor((self.feature_map_sz[0]-1)/2)),axis=0)
        self.M_prev = np.zeros(self.feature_map_sz)

        self.small_filter_sz=(int(np.floor(self.base_target_sz[0]/self.feature_ratio)),int(np.floor(self.base_target_sz[1]/self.feature_ratio)))

        # self.scale_estimator = LPScaleEstimator(self.target_sz, config=self.scale_config)
        # self.scale_estimator.init(image, self._center, self.base_target_sz, self.current_scale_factor)

        pixels=self.get_sub_window(image,self._center,model_sz=self.crop_size,
                                   scaled_sz=(int(np.round(self.crop_size[0]*self.current_scale_factor)),
                                              int(np.round(self.crop_size[1]*self.current_scale_factor))))
        feature=self.extract_hc_feture(pixels, cell_size=self.feature_ratio)
        self.model_xf=fft2(self._window[:,:,None]*feature)

        self.g_f=self.ADMM(self.model_xf)
        #self.model_xf = self.model_xf[:,:,:,np.newaxis]
        responsef = np.sum(np.conj(self.g_f)*self.model_xf,axis=2)
        if len(np.shape(responsef)) == 4:
            responsef = np.transpose(responsef,(0,1,3,2))
        # responsef = responsef[:,:,np.newaxis] 在函数中加维度
        responsef_padded = resize_dft2(responsef,self.interp_sz)
        # response in the spatial domain
        # MATLAB: response = ifft2(responsef_padded, 'symmetric')
        response = np.real(ifft2(responsef_padded))
        # response = np.real(cifft2(responsef_padded))
        # response = ifft2_sym(responsef_padded)
        self.M_prev = np.squeeze(np.fft.fftshift(response))
        # self.M_prev = np.fft.fftshift(response)
        # self.M_prev = np.sum(self.M_prev,axis=2)
        self.max_M_prev = self.M_prev.max()
        id_max_prev = np.argwhere(self.M_prev == self.max_M_prev)
        self.id_ymax_prev = id_max_prev[:,0]
        self.id_xmax_prev = id_max_prev[:,1]
        # target_sz= tuple(np.floor((self.target_sz[0]*self.current_scale_factor,self.target_sz[1]*self.current_scale_factor)))
        # new_state = [self._center[0]-np.floor(target_sz[0]/2),self._center[1]-np.floor(target_sz[1]/2),target_sz[0],target_sz[1]]
        # out = {'target_bbox': new_state}
        # return out

    def update(self, image, frame_id):
        self.frame = frame_id
        x=None
        for scale_ind in range(self.number_of_scales):
            current_scale=self.current_scale_factor*self.scale_factors[scale_ind]
            sub_window=self.get_sub_window(image,self._center,model_sz=self.crop_size,
                                        scaled_sz=(int(round(self.crop_size[0]*current_scale)),
                                    int(round(self.crop_size[1]*current_scale))))
            feature= self.extract_hc_feture(sub_window, self.cell_size)[:, :, :, np.newaxis]
            if x is None:
                x=feature
            else:
                x=np.concatenate((x,feature),axis=3)
        xtf=fft2(x*self._window[:,:,None,None])
        responsef=np.sum(np.conj(self.g_f)[:,:,:,None]*xtf,axis=2)
        if len(np.shape(responsef)) == 4:
            responsef = np.transpose(responsef,(0,1,3,2))
            
        if self.interpolate_response==2:
            self.interp_sz=(int(np.floor(self.yf.shape[1]*self.feature_ratio*self.current_scale_factor)),
                            int(np.floor(self.yf.shape[0]*self.feature_ratio*self.current_scale_factor)))
        responsef_padded=resize_dft2(responsef,self.interp_sz)
        response=np.real(ifft2(responsef_padded))
        # response = np.real(cifft2(responsef_padded))
        # response = ifft2_sym(responsef_padded)
        if self.interpolate_response==3:
            raise ValueError

        elif self.interpolate_response==4:
            disp_row,disp_col,sind=resp_newton(response,responsef_padded,self.newton_iterations, self.ky,self.kx,self.feature_map_sz)
        else:
            row,col,sind=np.unravel_index(np.argmax(response,axis=None),response.shape)
            disp_row = (row - 1 + int(np.floor(self.interp_sz[1] - 1) / 2)) % self.interp_sz[1] - int(np.floor((self.interp_sz[1] - 1) / 2))
            disp_col = (col - 1 + int(np.floor(self.interp_sz[0] - 1) / 2)) % self.interp_sz[0] - int(np.floor((self.interp_sz[0] - 1) / 2))

        if self.interpolate_response==0  or self.interpolate_response==3 or self.interpolate_response==4:
            factor=self.feature_ratio*self.current_scale_factor*self.scale_factors[sind]
        elif self.interpolate_response==1:
            factor=self.current_scale_factor*self.scale_factors[sind]
        elif self.interpolate_response==2:
            factor=self.scale_factors[sind]
        else:
            raise ValueError
        dx,dy=int(np.round(disp_col*factor)),int(np.round(disp_row*factor))
        self.current_scale_factor=self.current_scale_factor*self.scale_factors[sind]
        self.current_scale_factor=max(self.current_scale_factor,self.min_scale_factor)
        self.current_scale_factor=min(self.current_scale_factor,self.max_scale_factor)

        # self.current_scale_factor = self.scale_estimator.update(image, self._center, self.base_target_sz,
        #                                       self.current_scale_factor)

        self._center=(self._center[0]+dx,self._center[1]+dy)
        # find peak in the map
        self.M_curr = np.fft.fftshift(response[:,:,sind])
        self.max_M_curr = self.M_curr.max()
        id_max_curr = np.argwhere(self.M_curr == self.max_M_curr)
        # self.id_xmax_curr = id_max_curr[0][0]
        # self.id_ymax_curr = id_max_curr[0][1]
        self.id_ymax_curr = id_max_curr[:,0]
        self.id_xmax_curr = id_max_curr[:,1]

        # do shifting of previous response map
        shift_x = self.id_xmax_curr - self.id_xmax_prev
        shift_y = self.id_ymax_curr - self.id_ymax_prev
        sz_shift_y = len(shift_y)
        sz_shift_x = len(shift_x)
        if sz_shift_y > 1:
            shift_y = shift_y[0]
        if sz_shift_x > 1:
            shift_x = shift_x[0]

        self.M_prev = np.roll(self.M_prev,shift_y,0)
        self.M_prev = np.roll(self.M_prev,shift_x,1)

        # self.M_prev = np.roll(self.M_prev,shift_x,0)
        # self.M_prev = np.roll(self.M_prev,shift_y,1)

        # map difference
        # map_diff(frame) = norm(abs(M_prev - M_curr))

        pixels=self.get_sub_window(image,self._center,model_sz=self.crop_size,
                                   scaled_sz=(int(round(self.crop_size[0]*self.current_scale_factor)),
                                              int(round(self.crop_size[1]*self.current_scale_factor))))
        feature=self.extract_hc_feture(pixels, cell_size=self.cell_size)

        #feature=cv2.resize(pixels,self.feature_map_sz)/255-0.5
        xf=fft2(feature*self._window[:,:,None])

        ##
        self.model_xf=(1-self.interp_factor)*self.model_xf+self.interp_factor*xf
        self.g_f = self.ADMM(self.model_xf)

        self.M_prev = self.M_curr
        self.max_M_prev = self.max_M_curr
        self.id_xmax_prev = self.id_xmax_curr
        self.id_ymax_prev = self.id_ymax_curr
        
        
        # target_sz= tuple(np.floor((self.base_target_sz[0]*self.current_scale_factor,self.base_target_sz[1]*self.current_scale_factor)))
        target_sz=(self.base_target_sz[0]*self.current_scale_factor,self.base_target_sz[1]*self.current_scale_factor)
        # new_state = [self._center[0]-np.floor(target_sz[0]/2),self._center[1]-np.floor(target_sz[1]/2),target_sz[0],target_sz[1]]
        new_state = [self._center[0]-target_sz[0]/2,self._center[1]-target_sz[1]/2,target_sz[0],target_sz[1]]
        # out = {'target_bbox': new_state}
        box = np.array(new_state)
        return box

    def get_subwindow_no_window(self,img,pos,sz):
        h,w=sz[1],sz[0]
        xs = (np.floor(pos[0]) + np.arange(w) - np.floor(w / 2)).astype(np.int64)
        ys = (np.floor(pos[1]) + np.arange(h) - np.floor(h / 2)).astype(np.int64)
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= img.shape[1]] = img.shape[1] - 1
        ys[ys >= img.shape[0]] = img.shape[0] - 1
        out = img[ys, :][:, xs]
        xs,ys=np.meshgrid(xs,ys)
        return xs,ys,out

    def ADMM(self,xf):
        g_f = np.zeros_like(xf)
        h_f = np.zeros_like(g_f)
        l_f = np.zeros_like(g_f)
        mu = 1
        beta = 10
        mumax = 10000
        i = 1
        T = self.feature_map_sz[0] * self.feature_map_sz[1]
        S_xx = np.sum(np.conj(xf) * xf, 2)
        while i <= self.admm_iterations:
            A = mu / (self.admm_gamma + 1)
            B = S_xx + T * A
            S_lx = np.sum(np.conj(xf) * l_f, axis=2)
            S_hx = np.sum(np.conj(xf) * h_f, axis=2)
            # tmp0 = (1 / (T * A) * (self.yf[:, :, None] * xf)) + (self.admm_gamma / A) * (self.M_prev[:, :, None] * xf) - ((1 / A) * l_f) + (mu/A)*h_f
            # tmp1 = 1 / (T * A) * (xf * ((S_xx * self.yf)[:, :, None])) + (self.admm_gamma/A) * (xf * ((S_xx[:,:,np.newaxis] * self.M_prev[:, :, None])))
            tmp0 = ((1 / (T * A)) * (self.yf[:, :, None] * xf)) + (self.admm_gamma / A) * (self.M_prev[:,:,None] * xf) - ((1 / A) * l_f) + (mu/A)*h_f
            tmp1 = 1 / (T * A) * (xf * ((S_xx * self.yf)[:, :, None])) + (self.admm_gamma/A) * (xf * ((S_xx * self.M_prev)[:,:,None]))
            tmp2 = (1 / A) * (xf * (S_lx[:, :, None]))
            tmp3 = (mu/A) * xf * S_hx[:, :, None]
            # solve for g
            g_f = (1 / (1 + self.admm_gamma)) * (tmp0 - (tmp1 - tmp2 + tmp3) / B[:, :, None])
            # solve for h
            h = (T / ((mu * T) + self.admm_lambda)) * ifft2(mu * g_f + l_f)
            xs, ys, h = self.get_subwindow_no_window(h,
                                                     (np.floor(self.feature_map_sz[0] / 2), np.floor(self.feature_map_sz[1] / 2)),
                                                     self.small_filter_sz)
            t = np.zeros((self.feature_map_sz[1], self.feature_map_sz[0], h.shape[2]),dtype=np.complex64)
            # if len(np.shape(h)) == 4:
            #     h = np.sum(h,axis=3)
            t[ys,xs,:] = h
            # t[ys,:][:,xs] = h
            h_f = fft2(t)
            l_f = l_f + (mu * (g_f - h_f))
            mu = min(beta * mu, mumax)
            i += 1
        return g_f


    def get_sub_window(self, img, center, model_sz, scaled_sz=None):
        model_sz = (int(model_sz[0]), int(model_sz[1]))
        if scaled_sz is None:
            sz = model_sz
        else:
            sz = scaled_sz
        sz = (max(int(sz[0]), 2), max(int(sz[1]), 2))

        # without padding
        xs = (np.floor(center[0]) + np.arange(sz[0])+1 - np.floor(sz[0]/2)).astype(np.int64)-1
        ys = (np.floor(center[1]) + np.arange(sz[1])+1 - np.floor(sz[1]/2)).astype(np.int64)-1
        # %check for out-of-bounds coordinates, and set them to the values at
        # %the borders
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= img.shape[1]] = img.shape[1] - 1
        ys[ys >= img.shape[0]] = img.shape[0] - 1
        im_patch = img[ys, :][:, xs]
        # im_patch = cv2.getRectSubPix(img, sz, center)
        if model_sz is not None:
            im_patch = mex_resize(im_patch, model_sz)
        # if (min(xs)<1 and min(ys)<1) or (max(xs)>img.shape[1] and max(ys)>img.shape[0]):
        #     cv2.imwrite('test.jpg',im_patch.astype(np.uint8))
        return im_patch.astype(np.uint8)

    def extract_hc_feture(self,patch,cell_size):
        # patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        hog_feature=extract_hog_feature(patch,cell_size)
        # cn_feature=extract_cn_feature(patch,cell_size)
        # hc_feature=np.concatenate((hog_feature,cn_feature),axis=2)
        # return hc_feature
        return hog_feature



