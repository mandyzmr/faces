import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #在终端的输出
        self.file = None

    def open(self, file, mode='w'): #默认新建日志
        self.file = open(file, mode)

    def write(self, message, is_terminal=0, is_file=1):
        if is_terminal == 1:
            self.terminal.write(message) #在终端输出日志
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message) #在文件输出日志
            self.file.flush()


def time_to_str(t):
    hr = int(t/60)//60
    minute = int(t/60)%60
    sec = int(t)%60   
    return f'{hr:02}h {minute:02}m {sec:02}s'

