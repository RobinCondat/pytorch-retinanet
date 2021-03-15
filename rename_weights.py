import torch
from collections import OrderedDict

import argparse

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_model', help='Path of Model', default='',type=str)
    parser.add_argument('--suffix',help='Suffix to add to layers', default='', type=str)
    parser.add_argument('--path_save', help='Path of new model', default='',type=str)
    args = parser.parse_args(args)

    device = torch.device('cpu')
    model = torch.load(args.path_model, map_location=device)
    print("Init model keys : ")
    print(model.keys())

    if not args.suffix.startswith('_'):
        args.suffix = '_'+args.suffix+'.'
    else:
        args.suffix=args.suffix+'.'

    # Add suffix to backbone layers
    keys_to_change = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']
    model_2 = model
    for key in keys_to_change:
        model_2 = OrderedDict([(k.split('.')[0]+args.suffix+'.'.join(k.split('.')[1:]), v) if k.startswith(key) else (k, v) for k, v in model_2.items()])

    # Keep only backbone layers
    model_3 = OrderedDict([(k,v) for k, v in model_2.items() if args.suffix in k])
    print("New model keys : ")
    print(model_3.keys())
    torch.save(model_3,args.path_save)

if __name__ == '__main__':
    print('hey')
    main()