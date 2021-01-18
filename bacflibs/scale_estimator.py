import numpy as np
import cv2
from numpy.fft import fft, ifft
# from .feature import extract_hog_feature
from .feature import extract_pyhog_feature
from .utils import cos_window
from .fft_tools import ifft2,fft2

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
        # get scale filter features
        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz,self.window)

        # project
#        xs = self.basis.dot(xs) * self.window

        # get scores
        xsf = np.fft.fft(xs, axis=1)
        scale_responsef = np.sum(self.sf_num * xsf, 0) / (self.sf_den + self.config.scale_lambda)
#        interp_scale_response = np.real(ifft(resize_dft(scale_responsef, self.config.number_of_interp_scales)))
        recovered_scale_index = np.argmax(np.real(ifft(scale_responsef)))

#        if self.config.do_poly_interp:
#            # fit a quadratic polynomial to get a refined scale estimate
#            id1 = (recovered_scale_index - 1) % self.config.number_of_interp_scales
#            id2 = (recovered_scale_index + 1) % self.config.number_of_interp_scales
#            poly_x = np.array([self.interp_scale_factors[id1], self.interp_scale_factors[recovered_scale_index],
#                               self.interp_scale_factors[id2]])
#            poly_y = np.array(
#                [interp_scale_response[id1], interp_scale_response[recovered_scale_index], interp_scale_response[id2]])
#            poly_A = np.array([[poly_x[0] ** 2, poly_x[0], 1],
#                               [poly_x[1] ** 2, poly_x[1], 1],
#                               [poly_x[2] ** 2, poly_x[2], 1]], dtype=np.float32)
#            poly = np.linalg.inv(poly_A).dot(poly_y.T)
#            scale_change_factor = - poly[1] / (2 * poly[0])
#        else:
#        scale_change_factor = self.interp_scale_factors[recovered_scale_index]


        current_scale_factor=current_scale_factor*self.scale_size_factors[recovered_scale_index]

        scales = current_scale_factor * self.scale_size_factors
#        xs = self._shift_scale_sample(im, pos, base_target_sz, xs, recovered_scale_index, scales, self.window, self.scale_model_sz)
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz,self.window)
#        self.s_num = (1 - self.config.scale_learning_rate) * self.s_num + self.config.scale_learning_rate * xs
#        # compute projection basis
#        if self.max_scale_dim:
#            self.basis, _ = scipy.linalg.qr(self.s_num, mode='economic')
#            scale_basis_den, _ = scipy.linalg.qr(xs, mode='economic')
#        else:
#            U, _, _ = np.linalg.svd(self.s_num)
#            self.basis = U[:, :self.s_num_compressed_dim]
#            V,_,_=np.linalg.svd(xs)
#            scale_basis_den=V[:,:self.s_num_compressed_dim]
#        self.basis = self.basis.T
#
#        # compute numerator
#        feat_proj = self.basis.dot(self.s_num) * self.window
        xsf = np.fft.fft(xs, axis=1)
        new_sf_num = self.yf * np.conj(xsf)
        new_sf_den = np.sum(xsf*np.conj(xsf), 0)
        self.sf_num = (1 - self.config.scale_learning_rate) * self.sf_num + self.config.scale_learning_rate * new_sf_num
        self.sf_den = (1 - self.config.scale_learning_rate) * self.sf_den + self.config.scale_learning_rate * new_sf_den
        return current_scale_factor


    def _extract_scale_sample(self, im, pos, base_target_sz, scale_factors, scale_model_sz,window):
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
            temp=extract_pyhog_feature(im_patch_resized,cell_size=4)
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
                temp=extract_pyhog_feature(im_patch_resized,cell_size=4)
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
                temp=extract_pyhog_feature(im_patch_resized,cell_size=4)
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

class LPScaleEstimator:
    def __init__(self,target_sz,config):
        self.learning_rate_scale=config.learning_rate_scale
        self.scale_sz_window = config.scale_sz_window
        self.target_sz=target_sz

    def init(self,im,pos,base_target_sz,current_scale_factor):
        w,h=base_target_sz
        avg_dim = (w + h) / 2.5
        self.scale_sz = ((w + avg_dim) / current_scale_factor,
                         (h + avg_dim) / current_scale_factor)
        self.scale_sz0 = self.scale_sz
        self.cos_window_scale = cos_window((self.scale_sz_window[0], self.scale_sz_window[1]))
        self.mag = self.cos_window_scale.shape[0] / np.log(np.sqrt((self.cos_window_scale.shape[0] ** 2 +
                                                                    self.cos_window_scale.shape[1] ** 2) / 4))

        # scale lp
        patchL = cv2.getRectSubPix(im, (int(np.floor(current_scale_factor * self.scale_sz[0])),
                                                 int(np.floor(current_scale_factor * self.scale_sz[1]))), pos)
        patchL = cv2.resize(patchL, self.scale_sz_window)
        patchLp = cv2.logPolar(patchL.astype(np.float32), ((patchL.shape[1] - 1) / 2, (patchL.shape[0] - 1) / 2),
                               self.mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

        self.model_patchLp = extract_pyhog_feature(patchLp, cell_size=4)

    def update(self,im,pos,base_target_sz,current_scale_factor):
        patchL = cv2.getRectSubPix(im, (int(np.floor(current_scale_factor * self.scale_sz[0])),
                                                   int(np.floor(current_scale_factor* self.scale_sz[1]))),pos)
        patchL = cv2.resize(patchL, self.scale_sz_window)
        # convert into logpolar
        patchLp = cv2.logPolar(patchL.astype(np.float32), ((patchL.shape[1] - 1) / 2, (patchL.shape[0] - 1) / 2),
                               self.mag, flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        patchLp = extract_pyhog_feature(patchLp, cell_size=4)
        tmp_sc, _, _ = self.estimate_scale(self.model_patchLp, patchLp, self.mag)
        tmp_sc = np.clip(tmp_sc, a_min=0.6, a_max=1.4)
        scale_factor=current_scale_factor*tmp_sc
        self.model_patchLp = (1 - self.learning_rate_scale) * self.model_patchLp + self.learning_rate_scale * patchLp
        return scale_factor

    def estimate_scale(self,model,obser,mag):
        def phase_correlation(src1,src2):
            s1f=fft2(src1)
            s2f=fft2(src2)
            num=s2f*np.conj(s1f)
            d=np.sqrt(num*np.conj(num))+2e-16
            Cf=np.sum(num/d,axis=2)
            C=np.real(ifft2(Cf))
            C=np.fft.fftshift(C,axes=(0,1))

            mscore=np.max(C)
            pty,ptx=np.unravel_index(np.argmax(C, axis=None), C.shape)
            slobe_y=slobe_x=1
            idy=np.arange(pty-slobe_y,pty+slobe_y+1).astype(np.int64)
            idx=np.arange(ptx-slobe_x,ptx+slobe_x+1).astype(np.int64)
            idy=np.clip(idy,a_min=0,a_max=C.shape[0]-1)
            idx=np.clip(idx,a_min=0,a_max=C.shape[1]-1)
            weight_patch=C[idy,:][:,idx]

            s=np.sum(weight_patch)+2e-16
            pty=np.sum(np.sum(weight_patch,axis=1)*idy)/s
            ptx=np.sum(np.sum(weight_patch,axis=0)*idx)/s
            pty=pty-(src1.shape[0])//2
            ptx=ptx-(src1.shape[1])//2
            return ptx,pty,mscore

        ptx,pty,mscore=phase_correlation(model,obser)
        rotate=pty*np.pi/(np.floor(obser.shape[1]/2))
        scale = np.exp(ptx/mag)
        return scale,rotate,mscore

