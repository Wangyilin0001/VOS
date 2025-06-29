import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

# python test.py --which_epoch 200

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    line = 0
    epoch = 20

    data_loader,l1,l2,best_ls_x,d_ls = CreateDataLoader(opt,line,epoch,two=2)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)


    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))


    for i, data in enumerate(dataset):
        # if i >= opt.how_many:
        #     break
        model.set_input(data)
        model.test(l1,l2,best_ls_x,d_ls)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

