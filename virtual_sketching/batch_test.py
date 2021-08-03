from virtual_sketching.test_vectorization import *
import time


def main_self(model_base_dir, sketch_path, test_image_name, sampling_path, sampling_num, max_stroke=700):
    model_name = 'pretrain_clean_line_drawings'

    state_dependent = False
    longer_infer_lens = [max_stroke for _ in range(10)]
    round_stop_state_num = 12
    stroke_acc_threshold = 0.95

    draw_seq = False
    draw_color_order = True

    # set numpy output to something sensible
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

    main_testing(sketch_path, test_image_name, sampling_path, model_base_dir, model_name, sampling_num,
                 draw_seq=draw_seq, draw_order=draw_color_order,
                 state_dependent=state_dependent, longer_infer_lens=longer_infer_lens,
                 round_stop_state_num=round_stop_state_num, stroke_acc_threshold=stroke_acc_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='../data/dataset/sketch/', help="The test image name.")
    parser.add_argument('--output', '-o', type=str, default='../data/dataset/stroke/', help="The test image name.")
    parser.add_argument('--sample', '-s', type=int, default=5, help="The number of outputs.")
    parser.add_argument('--model_path', '-m', type=str, default='model/', help="The model direction")
    parser.add_argument('--max_stroke', type=int, default=700, help="The max stroke number for one round")
    args = parser.parse_args()

    # args.model_path = '../../virtual_sketching-main/outputs/snapshot/'

    assert args.input != ''
    assert args.sample > 0
    if not os.path.exists(args.output): os.makedirs(args.output)

    item_name_list = os.listdir(args.input)
    item_name_list.sort()
    num = len(item_name_list)
    resume = 0

    start_time = time.time()
    for i in range(resume, num):
        name = item_name_list[i]
        print('\n===========================================')
        print('process image {:d}/{:d}, {}'.format(i, num, item_name_list[i]))
        print('===========================================')
        print('Time cost so far: {:.3f}s'.format(time.time() - start_time))

        main_self(args.model_path, args.input, name, args.output, args.sample, args.max_stroke)
