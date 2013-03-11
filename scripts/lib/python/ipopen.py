#!/usr/bin/env python2.7

##################################################################
# Description
"""Module for interactive communication with sub-process pipes.

   Classes:
   IPopen() - interactive version of subprocess.Popen class.

   Exceptions:
   PipeTimeoutError - exception raised when pipe process doesn't
                      finish its output within specified time.

"""

##################################################################
# Modules
import subprocess
import sys
import threading

##################################################################
# Class
class IPopen(subprocess.Popen):

    """Interactive version of standard subprocess.Popen class.

    This class inherits most of its methods from its parent - the
    subprocess.Popen class.  Additionally, it extends or overrides some of the
    parental methods to provide a possibility of interactive communication with
    an opened pipe process. In contrast to the parent, a pipe sub-process isn't
    killed and re-instantiated every time, when new input data is passed to
    pipe. Instead, a conventional terminating string is passed after every
    contiguous block of input data and the output is only read until this
    terminating sequence is encountered. The pipe subprocess should be
    explicitly killed at the end, by invoking the close() method of a class's
    instance. Beware, a deadlock may occur, if terminating sequence gets
    modified or if piped sub-process uses buffering. To prevent this, an
    error will be raised after `timeout' seconds elapse after the last input if
    no output was received in that time.

    This class provides following instance variables:
    self.skip_line        - terminating sequence, which will be passed after
                            every block of input lines
    self.skip_line_expect - regexp for matching self.skip_line from pipe's output
    self.timeout          - timeout in second for waiting for output to be printed
    self.err              - exception raised by an underlying thread

    The following method(s) were extended:
    __init__()            - initialize the aforementioned instance variables and
                            call parent's constructor with stdin and stdout set to
                            subprocess.PIPE and `close_fds' argument set to true

    The following method(s) were overridden:
    self.communicate()    - pass input to pipe and return its output
    self.close()          - close file descriptors opened for pipe

    """

    def __init__(self, skip_line = '\n\n', skip_line_expect = '\n', \
                     timeout = 2, **kwargs):
        """Initialize instance variables and call parent's constructor.

        This method has following kw arguments from which eponymous instance
        variables are created:

        skip_line        - terminating sequence, which will be passed after
                           every block of input lines to pipe subprocess
        skip_line_expect - regexp for matching self.skip_line from pipe's
                           output
        timeout          - timeout in second for waiting for output to be
                           printed (defaults to 2)
        err              - an instance variables which is initially set to (),
                           which will hold an exception if one should be raised in
                           a thread

        all the remaining key-word arguments are passed unchagend to the
        constructor of parental class - subprocess.Popen().

        As in every regular __init__() method, the return value is void().

        """
        self.skip_line = skip_line
        self.skip_line_expect = skip_line_expect
        self.timeout = timeout
        self.err     = ()
        self.__output__ = ''
        super(IPopen, self).__init__(stdin = subprocess.PIPE, \
                                         stdout = subprocess.PIPE, \
                                         close_fds = True, \
                                         **kwargs)

    def communicate(self, input_txt, encd = 'utf-8'):
        """Pass the arguments to a helper function and return its output.

        This functions creates a separate thread for communicating with
        subprocess pipe and passes input_txt to subprocess via
        __communicate_helper_() function in this thread. So, if helper
        function doesn't finish within allocated time an exception is raised.

        """
        self.__output__ = ''
        thread = threading.Thread(target = self.__communicate_helper_, \
                                      args = (input_txt, encd))
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            self.close()
            raise PipeTimeoutError(self.timeout)
        else:
            # if thread completed within allocated time, but raised an
            # exception, re-raise this exception here
            if self.err:
                # unfortunately, simply unpacking `self.err' produces a syntax
                # error here
                raise self.err[0], self.err[1]
            return self.__output__

    def __communicate_helper_(self, input_txt, encd='utf-8'):
        """Pass input_txt to pipe and return its output.

        This method has following arguments:

        input_txt - Unicode text, which should be passed to internal pipe
                    sub-process
        encd      - encoding to which input_txt should be converted before
                    being passed to pipe (defaults to "utf-8")

        The ouput of this function is resulting text from the pipe. If the text
        wasn't obtained within self.timeout seconds counting from the end of
        input, a RuntimError will be raised.

        """
        input_txt = (input_txt + self.skip_line).encode(encd)
        output = ''
        try:
            self.stdin.write(input_txt)
            line = True
            while (not (self.stdout.closed and self.poll())) and line:
                line = self.stdout.readline().decode(encd)
                if line and line == self.skip_line_expect:
                    self.__output__ = output
                    return self.__output__
                else:
                    output += line
        # if any exception was raised during communication with pipe -
        # propagate this exception to the parent process
        except:
            self.err = sys.exc_info()
            raise

    def close(self):
        """Close file descriptors opened by subprocess pipe and check exit status.

        This methods closes stdin and stdout descriptors opened for pipe process and
        checks whether the exit status was 0. If not, a RuntimeError will be raised."""
        if not self.poll():
            # close stdin first, and hope that stdout will be closed
            # automatically
            if not self.stdin.closed:
                self.stdin.close()
            self.wait()
        if self.returncode != 0:
            raise RuntimeError('Abnormal exit of pipe command. Exit code: ' + \
                                   str(self.returncode))
        return None

##################################################################
# Exception
class PipeTimeoutError(Exception):
    """Exception raised when pipe's output times out.

    Extended methods:
    __init__(self, timeout)  - calls parent constructor with pre-defined message

    """

    def __init__(self, timeout):
        """Call parent's constructor with pre-specified message."""
        Exception.__init__(self, """\
Pipe process didn't yield its output within allocated time ({:d} sec).""".format(timeout))
