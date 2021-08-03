import numpy as np
import cv2 as cv
import imageio


# ===========================
# Global configure parameters
# ===========================
C_stroke_resolution = 40  # indicate use how many points to represent one stroke
C_line_mode = 'W'  # or 'U', 'W' represents strokes with width, 'U' represents uniform line width
C_line_width = 1  # valid when line mode == 'U', specify the straight line width
C_round_num = 10
# choose how many round to draw
C_min_stroke_len = 0.003  # minimal stroke length threshold, if smaller than this, do not draw
C_canvas_width = 600  # rendered canvas width, bigger size represents higher resolution

V_disp_stop_time = 50  # unit: ms


def normal(x, width):
    return int(x * (width - 1) + 0.5)


def euclid_distance(p1, p2):
    diff_x = p1[0] - p2[0]
    diff_y = p1[1] - p2[1]
    len = np.sqrt(diff_x**2 + diff_y**2)
    return len


def stroke_draw(f, steps=100, width=128):
    x_list = np.zeros([steps, ])
    y_list = np.zeros([steps, ])
    w_list = np.zeros([steps, ])

    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = int(1 + z0 * width // 2)
    z2 = int(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype('float32')
    tmp = 1. / steps

    for i in range(steps):
        t = i * tmp
        x = int((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
        y = int((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
        z = int((1 - t) * z0 + t * z2)  # line width
        w = (1 - t) * w0 + t * w2  # color
        cv.circle(canvas, (y, x), z, w, -1)

        x_list[i] = y/(2*width) - 0.5  # swap numpy order to image uv order
        y_list[i] = x/(2*width) - 0.5
        w_list[i] = z/(2*width)  # line width info

    stroke_len = 0
    for i in range(steps-1):
        tmp_len = euclid_distance((x_list[i], y_list[i]), (x_list[i+1], y_list[i+1]))
        stroke_len += tmp_len

    return x_list, y_list, w_list, stroke_len, 1 - cv.resize(canvas, dsize=(width, width))


def canvas_render(x_list, y_list, w_list=None, canvas_size=1000):
    assert len(x_list) == len(y_list), 'x and y point array length are not equal'
    point_num = len(x_list)
    canvas = np.zeros([canvas_size, canvas_size]).astype('float32')

    # If don't require width, use cv.line to render
    if w_list is None:
        for i in range(point_num-1):
            last_point = (int(round(x_list[i] * canvas_size)), int(round(y_list[i] * canvas_size)))
            curr_point = (int(round(x_list[i+1] * canvas_size)), int(round(y_list[i+1] * canvas_size)))
            cv.line(canvas, last_point, curr_point, 1, C_line_width)

    # If require width, use cv.circle to render
    else:
        for i in range(point_num):
            curr_point = (int(round(x_list[i] * canvas_size)), int(round(y_list[i] * canvas_size)))
            cv.circle(canvas, curr_point, int(round(canvas_size * w_list[i])), 1, -1)
            # cv.circle(canvas, curr_point, 3, 1, -1)

    return canvas


def canvas_draw(npz_path, min_window_size, raster_size, gif_name=None):
    # ==================
    # 1. load data
    # ==================

    data = np.load(npz_path, encoding='latin1', allow_pickle=True)
    strokes_data = data['strokes_data']  # [stroke num, 7]
    init_cursors = data['init_cursors']  # [round num, 2]
    image_size = data['image_size']  # scalar
    round_length = data['round_length']  # [round num, ]
    init_width = data['init_width']  # 0.01

    if round_length.ndim == 0:
        round_lengths = [round_length]
    else:
        round_lengths = round_length

    if init_cursors.ndim == 1:
        init_cursors = [init_cursors]

    print('round_lengths:', round_lengths)

    frames = []  # gif maker

    # ==================================================================
    # param data: (N_strokes, 9): flag, x0, y0, x1, y1, x2, y2, r0, r2
    # ==================================================================

    canvas = np.zeros((C_canvas_width, C_canvas_width), dtype=np.float32)  # [0.0-BG, 1.0-stroke]
    cursor_idx = 0

    # ====================
    # 2. start each round
    # ====================

    strokes_x_list = []
    strokes_y_list = []
    strokes_w_list = []
    strokes_length = []

    for round_idx in range(len(round_lengths)):
        # print('Making progress round', round_idx + 1, '/', len(round_lengths))
        round_length = round_lengths[round_idx]

        cursor_pos = init_cursors[cursor_idx]
        cursor_idx += 1

        prev_width = init_width
        prev_scaling = 1.0
        prev_window_size = float(raster_size)

        current_stroke_x_list = []
        current_stroke_y_list = []
        current_stroke_w_list = []
        current_strokes_length = 0

        # Each stroke or movement
        for round_inner_i in range(round_length):
            stroke_idx = np.sum(round_lengths[:round_idx]).astype(np.int32) + round_inner_i

            # 2.1 calculate current window size
            curr_window_size_raw = prev_scaling * prev_window_size
            curr_window_size_raw = np.maximum(curr_window_size_raw, min_window_size)
            curr_window_size_raw = np.minimum(curr_window_size_raw, image_size)

            # 2.2 read in current stroke parameters
            pen_state = strokes_data[stroke_idx][0]  # 0 for down, 1 for lifting
            stroke_params = strokes_data[stroke_idx, 1:]  # (6)

            x1y1, x2y2, width2, scaling2 = stroke_params[0:2], stroke_params[2:4], stroke_params[4], stroke_params[5]
            x0y0 = np.zeros_like(x2y2)
            x0y0 = np.divide(np.add(x0y0, 1.0), 2.0)  # convert to image coord, center is at left-up corner
            x2y2 = np.divide(np.add(x2y2, 1.0), 2.0)  # range: [0.0, 1.0]
            widths = np.stack([prev_width, width2], axis=0)
            # widths = np.stack([0.1*(20/curr_window_size), 0.1*(20/curr_window_size)], axis=0)  # (2)

            stroke_params_proc = np.concatenate([x0y0, x1y1, x2y2, widths], axis=-1)  # (8)

            # 2.3 propagate to next stroke parameters
            next_width = stroke_params[4]
            next_scaling = stroke_params[5]
            next_window_size = next_scaling * curr_window_size_raw
            next_window_size = np.maximum(next_window_size, min_window_size)
            next_window_size = np.minimum(next_window_size, image_size)

            # line width: need to consider the line width in different window scaling
            prev_width = next_width * (curr_window_size_raw / next_window_size)
            prev_scaling = next_scaling
            prev_window_size = curr_window_size_raw

            f = stroke_params_proc.tolist()
            f += [1.0, 1.0]  # color param
            x_list, y_list, w_list, tmp_length, gt_stroke_img = stroke_draw(f, C_stroke_resolution)

            # 2.4 convert stroke position from window coord to canvas coord
            if pen_state == 0:
                window2image_scaling = curr_window_size_raw / image_size
                x_list = window2image_scaling * x_list + cursor_pos[0]
                y_list = window2image_scaling * y_list + cursor_pos[1]
                w_list = window2image_scaling * w_list
                current_strokes_length += window2image_scaling * tmp_length
                current_stroke_x_list += x_list.tolist()
                current_stroke_y_list += y_list.tolist()
                current_stroke_w_list += w_list.tolist()

            else:  # when lift, render the last stroke

                # If the length of current stroke is reasonable
                if current_strokes_length > C_min_stroke_len:
                    # print('strokes_len', current_strokes_length)

                    # render and display
                    if C_line_mode == 'U':
                        render_img = canvas_render(current_stroke_x_list, current_stroke_y_list, None, C_canvas_width)
                    elif C_line_mode == 'W':
                        render_img = canvas_render(current_stroke_x_list, current_stroke_y_list, current_stroke_w_list, C_canvas_width)
                    else:
                        raise ValueError('Please check your "C_line_mode" code, cannot be recognized')

                    canvas = cv.bitwise_or(render_img, canvas)
                    show_canvas = 1 - canvas  # inverse image to show, white back and black drawing
                    frames.append(np.array(255*cv.resize(show_canvas, (300, int(300/show_canvas.shape[1]*show_canvas.shape[0]))), np.uint8))
                    # cv.imshow('canvas', show_canvas)
                    # cv.waitKey(V_disp_stop_time)

                    # restore in global variables
                    strokes_x_list.append(current_stroke_x_list)
                    strokes_y_list.append(current_stroke_y_list)
                    strokes_w_list.append(current_stroke_w_list)
                    strokes_length.append(current_strokes_length)

                # else:
                #     print('This stroke should be printed out')

                # Reset data
                current_stroke_x_list = []
                current_stroke_y_list = []
                current_stroke_w_list = []
                current_strokes_length = 0

            # 2.5 update cursor_pos based on hps.cursor_type
            new_cursor_offsets = stroke_params[2:4] * (float(curr_window_size_raw) / 2.0)  # (1, 6), patch-level
            new_cursor_offset_next = new_cursor_offsets

            # important!!!
            new_cursor_offset_next = np.concatenate([new_cursor_offset_next[1:2], new_cursor_offset_next[0:1]], axis=-1)

            cursor_pos_large = cursor_pos * float(image_size)

            stroke_position_next = cursor_pos_large + new_cursor_offset_next  # (2), large-level

            cursor_pos_large = stroke_position_next  # (2), large-level

            cursor_pos_large = np.minimum(np.maximum(cursor_pos_large, 0.0), float(image_size - 1))  # (2), large-level
            cursor_pos = cursor_pos_large / float(image_size)
        if round_idx == C_round_num - 1:
            break

    if gif_name:
        # add few more frames to create a short final display
        final_canvas = np.array(255 * cv.resize(1 - canvas, (300, int(300 / canvas.shape[1] * canvas.shape[0]))), np.uint8)
        for i in range(int(0.2*len(frames))):
            frames.append(final_canvas)
        imageio.mimsave(gif_name, frames, 'GIF', duration=V_disp_stop_time/(2*1000))

    canvas = np.array(255 * (1 - canvas), np.uint8)
    return strokes_x_list, strokes_y_list, strokes_w_list, strokes_length, canvas


# visualize vector file (.npz) and save them
def visualize_vec_file(file_pth, file_name, gif=True):
    min_window_size = 32
    raster_size = 128
    # create save names
    npz_path = file_pth + 'seq_data/' + file_name + '_0.npz'
    gif_path = None if not gif else file_pth + file_name + '.gif'

    # do the canvas draw
    _, _, _, _, canvas = canvas_draw(npz_path, min_window_size, raster_size, gif_path)
    cv.imwrite(file_pth + file_name + '_vector.png', canvas)


if __name__ == '__main__':
    npz_path = 'E:/Postgraduate/virtual_sketching-main/outputs/sampling/' \
               'clean_line_drawings__pretrain_clean_line_drawings/seq_data/0000_0.npz'

    npz_path = '../output/Pipeline_and_vectorize/seq_data/IMG_1060_sketch_0.npz'
    min_window_size = 32
    raster_size = 128

    strokes_x_list, strokes_y_list, strokes_w_list, strokes_length, canvas = canvas_draw(npz_path, min_window_size, raster_size, '../output/seq_data/sketch.gif')

    cv.imshow('canvas', canvas)
    cv.waitKey(0)


