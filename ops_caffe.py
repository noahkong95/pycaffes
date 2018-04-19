#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import numpy as np

caffe_root = './'
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P

caffe.set_mode_cpu()


def error_no():
    print "error :",__file__,sys._getframe().f_lineno

class op_caffe:
    def __init__(self, op_name='conv'):
        self.op_name = op_name;
        self.so = cdll.LoadLibrary('cvs.dll');

    def preprocess(self, im, mean, std):
        h,w,c = im.shape
        x = im.astype(np.float32);

        for i in xrange(c):
            x[:,:,i] = (x[:,:,i] - mean[i]) / float(std);

        return x;

    def convert_strideh(self, im):
        h,w,c = im.shape

        #if c == 3:
        #    x = np.zeros((),)

        return im;

    def resize_img(self, im, iw, ih, format, dst_size, dst_im, sbits = 15):
            

    def permute(self, data, input = 'CHW', output = 'HWC'):
        if (input == 'CHW') and (output == 'HWC'):
            data = np.transpose(data, (1, 0, 2));  # hcw
            out = np.transpose(data, (0, 2, 1));  # hwc
        elif (input == 'HWC') and (output == 'CHW'):
            data = np.transpose(data, (0, 2, 1));  # hcw
            out = np.transpose(data, (1, 0, 2));  # CHW
        elif (input == 'HWC') and (output == 'WHC'):
            out = np.transpose(data, (1, 0, 2));  # hcw
        elif (input == 'WHC') and (output == 'CHW'):
            data = np.transpose(data, (0, 2, 1));  # wch
            out1 = np.transpose(data, (1, 0, 2));  # cwh
            out = np.transpose(out1, (0, 2, 1));  # chw
        elif (input == 'CHW') and (output == 'WHC'):
            data = np.transpose(data, (0, 2, 1));  # cwh
            out1 = np.transpose(data, (1, 0, 2));  # wch
            out = np.transpose(out1, (0, 2, 1));  # whc
        else:
            return data;

        return out;

    def eltsum(self, data0, data1,coeff):

        if coeff:
            data = data0 * coeff[0] + data1 * coeff[1];
        else:
            data = data0 + data1;

        return data;
    
    def conv_net_sw(self,in_data, weight, bais, w=32,h=32,cin=32,cout=32,kw=3,kh=3,sw=1,sh=1,pw=1,ph=1,dw=1):
        cin,h,w = in_data.shape;
 
        model_path = 'temp/';
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        n = caffe.NetSpec();
        n.data0 = L.Input(shape=[dict(dim=[1, cin, h, w])])
        n.out = L.Convolution(n.data0, num_output=int(cout), kernel_w=int(kw), kernel_h=int(kh), stride_w=int(sw), stride_h=int(sh), pad_w=int(pw), pad_h=int(ph), dilation = int(dw));
        def_file = model_path + 'internal.prototxt'
        with open(def_file, 'w') as f:
            f.write(str(n.to_proto()));
            f.close()
        net = caffe.Net(def_file, caffe.TEST);

        pw = np.float32(weight.reshape(net.params['out'][0].data.shape));
        pb = np.float32(bais.reshape(net.params['out'][1].data.shape));
        net.params['out'][0].data[:] = pw;
        net.params['out'][1].data[:] = pb;

        in_data = np.float32(in_data.reshape([1, cin, h, w]));
        p = in_data

        net.blobs['data0'].data[...] = p
        output = net.forward()

        pa = np.float32(output['out'][0]);

        if not os.path.exists(model_path):
            os.remove(model_path)

        return pa;

    def mlp_net(self,in_data, weight, bais, cout = 32, axis = 1):
        cin,h,w = in_data.shape;
        
        model_path = 'temp/';
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        n = caffe.NetSpec();
        n.data0 = L.Input(shape=[dict(dim=[1, cin, h, w])])
        n.out = L.InnerProduct(n.data0,num_output = int(cout),axis = int(axis));
        def_file = model_path + 'internal.prototxt'
        with open(def_file, 'w') as f:
            f.write(str(n.to_proto()));
            f.close()
        net = caffe.Net(def_file, caffe.TEST);

        pw = np.float32(weight.reshape(net.params['out'][0].data.shape));
        pb = np.float32(bais.reshape(net.params['out'][1].data.shape));
        net.params['out'][0].data[:] = pw;
        net.params['out'][1].data[:] = pb;

        in_data = np.float32(in_data.reshape([1, cin, h, w]));
        p = in_data

        net.blobs['data0'].data[...] = p
        output = net.forward()
        pa = np.float32(output['out'][0]);
        
        if not os.path.exists(model_path):
            os.remove(model_path)

        return pa;
        
    def bn_net(self,in_data, weight, bais):
        cin,h,w = in_data.shape;
        model_path = 'temp/';
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        n = caffe.NetSpec();
        n.data0 = L.Input(shape=[dict(dim=[1, cin, h, w])])
        n.out = L.Scale(n.data0,bias_term=True);
        def_file = model_path + 'internal.prototxt'
        with open(def_file, 'w') as f:
            f.write(str(n.to_proto()));
            f.close()
        net = caffe.Net(def_file, caffe.TEST);

        pw = np.float32(weight.reshape(net.params['out'][0].data.shape));
        pb = np.float32(bais.reshape(net.params['out'][1].data.shape));
        net.params['out'][0].data[:] = pw;
        net.params['out'][1].data[:] = pb;

        in_data = np.float32(in_data.reshape([1, cin, h, w]));
        p = in_data

        net.blobs['data0'].data[...] = p
        output = net.forward()
        pa = np.float32(output['out'][0]);

        if not os.path.exists(model_path):
            os.remove(model_path)

        return pa;
    
    def relu_net(self,in_data, weight, prelu_type = False, channel_shared=False):
        cin,h,w = in_data.shape;
        model_path = 'temp/';
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        n = caffe.NetSpec();
        n.data0 = L.Input(shape=[dict(dim=[n1, cin, h, w])])
        if prelu_type:
            n.out = L.PReLU(n.data0,channel_shared = channel_shared);
        else:
            n.out = L.ReLU(n.data0);

        def_file = model_path + 'internal.prototxt'
        with open(def_file, 'w') as f:
            f.write(str(n.to_proto()));
            f.close()
        net = caffe.Net(def_file, caffe.TEST);

        in_data = np.float32(in_data.reshape([1, cin, h, w]));
        p = in_data

        if prelu_type:
            pw = np.float32(weight.reshape(net.params['out'][0].data.shape));
            net.params['out'][0].data[:] = pw;
        
        net.blobs['data0'].data[...] = p
        output = net.forward()
        pa = np.float32(output['out'][0]);
        
        if not os.path.exists(model_path):
            os.remove(model_path)

        return pa;
        
    def pool_net(self,in_data, pool = 0, kw = 3, kh = 3, sw = 1, sh = 1, pw = 1, ph = 1):
        cin,h,w = in_data.shape;
        model_path = 'temp/';
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        n = caffe.NetSpec();
        n.data0 = L.Input(shape=[dict(dim=[n1, cin, h, w])])
        n.out = L.Pooling(n.data0, pool = pool, kernel_w=int(kw), kernel_h=int(kh), stride_w=int(sw), stride_h=int(sh), pad_w=int(pw), pad_h=int(ph));
        def_file = model_path + 'internal.prototxt'
        with open(def_file, 'w') as f:
            f.write(str(n.to_proto()));
            f.close()
        net = caffe.Net(def_file, caffe.TEST);

        in_data = np.float32(in_data.reshape([1, cin, h, w]));
        p = in_data

        net.blobs['data0'].data[...] = p
        output = net.forward()
        pa = np.float32(output['out'][0]);

        if not os.path.exists(model_path):
            os.remove(model_path)

        return pa;

    def mute_net(self,in_data, order):
        cin,h,w = in_data.shape;
        model_path = 'temp/';
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        n = caffe.NetSpec();
        n.data0 = L.Input(shape=[dict(dim=[n1, cin, h, w])])
        n.out = L.Permute(n.data0, order=order);
        def_file = model_path + 'internal.prototxt'
        with open(def_file, 'w') as f:
            f.write(str(n.to_proto()));
            f.close()
        net = caffe.Net(def_file, caffe.TEST);

        in_data = np.float32(in_data.reshape([1, cin, h, w]));
        p = in_data

        net.blobs['data0'].data[...] = p
        output = net.forward()
        pa = np.float32(output['out'][0]);

        if not os.path.exists(model_path):
            os.remove(model_path)

        return pa;


"""
cin = 32;
cout = cin;
w = 32;
h = 32;

kw = 3;
kh = 3;
pw = 1
ph = 1
sw = 2
sh = 2

input = (np.random.random_sample([1*w*h*cin]) - 0.5);
p = np.float32(input.reshape([1,w,h,cin]));

pwt = (np.random.random_sample([cout]) - 0.5);
pwt = np.float32(pwt.reshape([1,cout]));
pb = (np.random.random_sample([1*cout]) - 0.5);
pb = np.float32(pb.reshape([1,cout]));

ops = op_caffe();

in_data = p
weight = pwt
bais = pb

bnout = ops.bn_net(in_data, weight, bais)
rlout = ops.relu_net(bnout, 0, prelu_type = False, channel_shared=False)
plout = ops.pool_net(rlout, pool = 0, kw = kw, kh = kh, sw = sw, sh = sh, pw = pw, ph = ph)

print plout.shape
"""

