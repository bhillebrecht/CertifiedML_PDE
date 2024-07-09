class State: 
    def __init__(self):
        self.d = 0.0
        self.t = 0.0
        self.old_dirs = []
        self.old_stps = []
        self.Hdiag = 1
        self.g_old = 0
        self.f_old = 0
        self.funcEval = 0
        self.nIter = 0

    def __init__(self, state):
        self.d = state.d
        self.t = state.t
        self.old_dirs = state.old_dirs
        self.old_stps = state.old_stps
        self.Hdiag = state.Hdiag
        self.g_old = state.g_old
        self.f_old = state.f_old
        self.funcEval = state.funcEval
        self.nIter = state.nIter
