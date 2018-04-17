"""
Initialize network
"""
# from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.vgg import vgg
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.alexnet import alexnet


def init_net(is_train, imdb_classes, args):
    """
    initilize the network here.
    """
    if args.net == 'alexnet':
        fasterRCNN = alexnet(imdb_classes, pretrained=is_train,
                             class_agnostic=args.class_agnostic)
    elif args.net == 'vgg11':
        fasterRCNN = vgg(imdb_classes, 11, pretrained=is_train,
                         class_agnostic=args.class_agnostic)
    elif args.net == 'vgg13':
        fasterRCNN = vgg(imdb_classes, 13, pretrained=is_train,
                         class_agnostic=args.class_agnostic)
    elif args.net == 'vgg16':
        # fasterRCNN = vgg16(imdb_classes, pretrained=is_train, class_agnostic=args.class_agnostic)
        fasterRCNN = vgg(imdb_classes, 16, pretrained=is_train,
                         class_agnostic=args.class_agnostic)
    elif args.net == 'vgg19':
        fasterRCNN = vgg(imdb_classes, 19, pretrained=is_train,
                         class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb_classes, 101, pretrained=is_train,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb_classes, 50, pretrained=is_train,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb_classes, 152, pretrained=is_train,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res18':
        fasterRCNN = resnet(imdb_classes, 18, pretrained=is_train,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res34':
        fasterRCNN = resnet(imdb_classes, 34, pretrained=is_train,
                            class_agnostic=args.class_agnostic)
    else:
        raise Exception("network is not defined")

    fasterRCNN.create_architecture()
    return fasterRCNN
