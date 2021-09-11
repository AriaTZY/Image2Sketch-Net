# Responsible for the tensorboard-like shape, write the decline curve and so on into the csv file
import os
import csv
import matplotlib.pylab as plt


class CSVManager:
    # 1）name 不需要加后缀名；2）head不需要加epoch和iter，这是默认的，只需要加想添加的值就行
    def __init__(self, name, root_path='tensorboard/csv/', head=None):
        self.root_path = root_path
        self.name = name
        self.csv_name = self.root_path + '/' + name + '.csv'
        if head is None:
            self.head = ['epoch', 'global_iter', 'loss']
        else:
            self.head = ['epoch', 'global_iter'] + head
        self.data_len = len(self.head)
        self.resume_epoch = 0
        self.resume_iter = 0
        self.line_num = 0
        os.makedirs(self.root_path, exist_ok=True)

        if os.path.exists(self.csv_name) and self.get_last_info():
            self.get_last_info()
            print('Keep writing logs from epoch {}, global iter {} ...'.format(self.resume_epoch, self.resume_iter))
        else:
            print('Create log file {}'.format(self.csv_name))
            with open(self.csv_name, 'w', newline='') as f:  # write head
                f_csv = csv.writer(f)
                f_csv.writerow(self.head)

    # get the info of resume point (last time)
    def get_last_info(self):
        with open(self.csv_name, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:  # only head line
                return False
            last_line = list(map(float, lines[-1].strip().split(',')))
            self.line_num = len(lines)
            self.resume_epoch = int(last_line[0])
            self.resume_iter = int(last_line[1])
            del lines
            return True

    def write_data(self, epoch, global_iter, dat):
        assert len(dat) == self.data_len - 2, 'The data you try to write is not identical to head'
        write_line = [self.resume_epoch+epoch, self.resume_iter+global_iter] + dat
        with open(self.csv_name, 'a+', newline="") as f:
            f_csv = csv.writer(f)
            f_csv.writerow(write_line)
            self.line_num += 1
        print('write new line done, global iter {}, line {}'.format(self.resume_iter+global_iter, self.line_num))

    def read_data(self):
        if os.path.exists(self.csv_name):
            with open(self.csv_name, 'r') as f:
                lines = f.readlines()
                if len(lines) <= 1:  # only head line
                    print('WARNING: There is only 1 or less lines here, cannot read!!!!')
                    return
                head = list(map(str, lines[0].strip().split(',')))
                data_dim = len(head)
                data_stack = [[] for _ in range(data_dim)]
                for i in range(1, len(lines)):
                    tmp_line = list(map(float, lines[i].strip().split(',')))
                    data_stack[0].append(int(tmp_line[0]))
                    data_stack[1].append(int(tmp_line[1]))
                    for col in range(2, data_dim):
                        data_stack[col].append(tmp_line[col])
                return head, data_stack

        else:
            print("File {} doesn't exist".format(self.csv_name))
            return

    # remember, "pick_cols" is a list, start from data col_idx
    def generate_graph(self, pick_cols):
        plt.cla()
        head, data_stack = self.read_data()
        iter_ls = data_stack[1]
        for i in range(1): #(len(pick_cols)):
            pick_now = pick_cols[i] + 2
            data_col = data_stack[pick_now]
            plt.plot(iter_ls, data_col, label=head[pick_now])
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('The loss curve of {}'.format(self.name))
        plt.legend()
        plt.savefig('loss_curve.png')
        plt.show()


if __name__ == '__main__':
    csv_log = CSVManager('deep_GAN_all')
    csv_log.generate_graph([0, 1, 2])
    # head, data = csv_log.read_data()
    # print(head)
    # print(data)
    # csv_log.write_data(0, 1, [3])
    # csv_log.write_data(0, 2, [3.3])
    # csv_log.write_data(0, 3, [1.0])


