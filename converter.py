from __future__ import print_function
import caffe  # NVCaffe
from os import path, getcwd
from caffe.proto import caffe_pb2
from argparse import ArgumentParser
import google.protobuf.text_format as txtf


#########################
##   CHECK FUNCTIONS   ##
#########################
def check_args(args):
    if not path.isfile(args.model):
        print('Model file ' + args.model + 'doesn\'t exist.')
        exit(-1)
    if args.weights != '' and not path.isfile(args.weights):
        print('Weights file ' + args.weights + 'doesn\'t exist.')
        exit(-1)
    if args.save_dir == '':
        args.save_dir = getcwd()
        print('You don\'t specify the --save_dir, outputs will be saved into the working dir: ' + args.save_dir)
    elif not path.isdir(args.save_dir):
        print('The save_dir you specify: ' + args.save_dir + ' is not a directory.')
        print('Please re-specify the --save_dir parameter.')
        exit(-1)
    print('All path checked!')


def generate_save_path(args, mode):
    assert mode in ['bvlc', 'merged']
    
    _, dst_model = path.split(args.model.replace('.prototxt', '_%s.prototxt' % mode))
    dst_model = path.join(args.save_dir, dst_model)
    _, dst_weights = path.split(args.weights.replace('.caffemodel', '_%s.caffemodel' % mode))
    dst_weights = path.join(args.save_dir, dst_weights)
    
    return dst_model, dst_weights


###################
##   CONVERTER   ##
###################
def generate_bn_scale(layer, layers, nv_bn_names):
    bn_param = layer.batch_norm_param
    if bn_param.HasField('use_global_stats'):
        bn_param.ClearField('use_global_stats')

    if bn_param.HasField('scale_filler'):
        bn_param.ClearField('scale_filler')
    if bn_param.HasField('bias_filler'):
        bn_param.ClearField('bias_filler')

    if bn_param.HasField('scale_bias'):
        if bn_param.scale_bias:
            bn_param.ClearField('scale_bias')
            layers.append(layer)
            scale_layer = caffe_pb2.LayerParameter()
            scale_layer.name = layer.name + '_scale'
            scale_layer.type = 'Scale'
            scale_layer.bottom.append(layer.top[0])
            scale_layer.top.append(layer.top[0])
            scale_layer.scale_param.filler.value = 1
            scale_layer.scale_param.bias_term = True
            scale_layer.scale_param.bias_filler.value = 0
            layers.append(scale_layer)
            nv_bn_names.append(layer.name)
        else:
            bn_param.ClearField('scale_bias')
            layers.append(layer)
    else:
        layers.append(layer)
        
        
def generate_bvlc_prototxt(src_model, dst_model):
    nv_net = caffe_pb2.NetParameter()
    bvlc_net = caffe_pb2.NetParameter()
    with open(src_model) as f:
        txtf.Merge(f.read(), nv_net)
    
    bvlc_net.name = nv_net.name + '_bvlc'
    layers = []
    nv_bn_names = []
    
    for layer in nv_net.layer:
        if not layer.type == 'BatchNorm':
            layers.append(layer)
        else:
            if layer.HasField('batch_norm_param'):
                generate_bn_scale(layer, layers, nv_bn_names)
            else:
                layers.append(layer)

    bvlc_net.layer.extend(layers)
    
    
    with open(dst_model, 'w') as f:
        f.write(str(bvlc_net))

    return nv_bn_names


def generate_bvlc_weights(src_model, src_weights, dst_model, dst_weights, nv_bn_names):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    nv_net = caffe.Net(src_model, src_weights, caffe.TEST)
    bvlc_net = caffe.Net(dst_model, caffe.TEST)
    
    for param in nv_net.params.keys():
        if param not in nv_bn_names:
            for i in xrange(len(nv_net.params[param])):
                bvlc_net.params[param][i].data[...] = nv_net.params[param][i].data
        else:
            bvlc_net.params[param][0].data[...] = nv_net.params[param][0].data
            bvlc_net.params[param][1].data[...] = nv_net.params[param][1].data
            bvlc_net.params[param][2].data[...] = nv_net.params[param][2].data
            bvlc_net.params[param + '_scale'][0].data[...] = nv_net.params[param][3].data
            bvlc_net.params[param + '_scale'][1].data[...] = nv_net.params[param][4].data

    bvlc_net.save(dst_weights)


##################
##   MERGE BN   ##
##################
def generate_merged_prototxt(src_model, dst_model):
    with open(src_model) as f:
        net = caffe_pb2.NetParameter()
        dst_net = caffe_pb2.NetParameter()
        txtf.Merge(f.read(), net)
    
    dst_net.name = net.name + '_merged'
    dst_layers = []
    change_bottom = False
    
    for i in range(len(net.layer)):
        layer = net.layer[i]
        if layer.type != 'BatchNorm' and layer.type != 'Scale':
            dst_layers.append(layer)
        if layer.type == 'Convolution':
            if net.layer[i+1].type == 'BatchNorm' and net.layer[i+2].type == 'Scale':
                if layer.convolution_param.bias_term == False:
                    layer.convolution_param.bias_term = True
                layer.top[0] = net.layer[i+2].top[0]
                i += 4
    
    dst_net.layer.extend(dst_layers)
    
    with open(dst_model, 'w') as f:
        f.write(str(dst_net))
    
    
def get_layer_name_by_index(net, index):
    return net.layer[index], net.layer[index].name
    
    
def generate_merged_weights(src_model, src_weights, dst_model, dst_weights):
    net = caffe.Net(src_model, src_weights, caffe.TEST)
    dst_net = caffe.Net(dst_model, caffe.TEST)
    with open(src_model) as f:
        net_param = caffe_pb2.NetParameter()
        txtf.Merge(f.read(), net_param)
    
    for i in range(len(net_param.layer)):
        layer, name = get_layer_name_by_index(net_param, i)
        if layer.type == 'Convolution':
            # extract w & b in convolution layer
            w = net.params[name][0].data
            batchsize = w.shape[0]
            try:
                b = net.params[name][1].data
            except:
                b = np.zeros(batchsize)
                
            # extract mean and var in BN layer
            layer, name = get_layer_name_by_index(net_param, i+1)
            mean = net.params[name][0].data
            var = net.params[name][1].data
            scalef = net.params[name][2].data
            if scalef != 0:
                scalef = 1. / scalef
            mean = mean * scalef
            var = var * scalef
            
            layer, name = get_layer_name_by_index(net_param, i+2)
            gamma = net.params[name][0].data
            beta = net.params[name][1].data
            
            # merge
            tmp = gamma/np.sqrt(var+1e-5)
            w = np.reshape(tmp, (batchsize, 1, 1, 1))*w
            b = tmp*(b-mean)+beta
            
            # store weight and bias in destination net
            layer, name = get_layer_name_by_index(net_param, i)
            dst_net.params[name][0].data[...] = w
            dst_net.params[name][1].data[...] = b
            
    dst_net.save(dst_weights)
    

#######################
##   MAIN FUNCTION   ##
#######################
def main(args):
    check_args(args)
    prototxt_only = args.weights == ''
    
    if args.convert:
        bvlc_model, bvlc_weights = generate_save_path(args, 'bvlc')
        nv_bn_names = generate_bvlc_prototxt(args.model, bvlc_model)
        print('\n================ Converting complete successfully. ===============')
        print('Save bvlc model (.prototxt) into: ' + bvlc_model)
        if not prototxt_only:
            generate_bvlc_weights(src_model, src_weights, bvlc_model, bvlc_weights, nv_bn_names)
            print('Save bvlc weights (.caffemodel) into: ' + bvlc_weights)
        print('================              Done.                ===============\n')
    
    if args.merge_bn:
        merged_model, merged_weights = generate_save_path(args, 'merged')
        src_model = bvlc_model if args.convert else args.model
        src_weights = bvlc_weights if args.convert else args.weights
        generate_merged_prototxt(src_model, merged_model)
        print('\n================ BN merging complete successfully. ===============')
        print('Save merged model (.prototxt) into: ' + merged_model)
        if not prototxt_only:
            generate_merged_weights(src_model, src_weights, merged_model, merged_weights)
            print('Save merged weights (.caffemodel) into: ' + merged_weights)
        print('================              Done.                ===============\n')


if __name__ == '__main__':
    parser = ArgumentParser(description='Model convertor. Both save NVCaffe model and weights to BVLC format and merge BN layer')
    parser.add_argument('--model', type=str, default='', help='The perseus-caffe model *.prototxt.')
    parser.add_argument('--weights', type=str, default='', help='The perseus-caffe weights *.caffemodel.')
    # parser.add_argument('--prototxt_only', type=bool, default=False, help='Only convert the prototxt.')
    parser.add_argument('--convert', type=bool, default=True, help='Convert NVCaffe model and weights to BVLC format')
    parser.add_argument('--merge_bn', type=bool, default=True, help='Merge BN layer into convolution.')
    parser.add_argument('--save_dir', type=str, default='', help='The directory to store the output.')
    args = parser.parse_args()
    main(args)
