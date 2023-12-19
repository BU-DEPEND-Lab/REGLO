from NeurNet import NeurNet

class SplitConfig:
    def __init__(self, config = None):
        self.split = {}
        self.hash = 0
        if config is not None:
            self.copy(config)
        self.next_candidate = None

    def copy(self, config):
        self.hash = config.hash
        for layer in config.split:
            for node in config.split[layer]:
                self.add(layer, node, config.split[layer][node])
    
    def add(self, layer, node, split):
        if layer not in self.split:
            self.split[layer] = {}
        if node in self.split[layer]:
            return False
        self.split[layer][node] = split
        max_n_nodes = 1_000_000_000_000
        if node >= max_n_nodes:
            raise(Exception(f"# of nodes in the layer exceed the supported maximum: {max_n_nodes}"))
        self.hash += (layer*max_n_nodes + node + 1) * 2 + split

    def setConfig(self, NN: NeurNet, batch_id = 0):
        for layer in self.split:
            for node in self.split[layer]:
                split = self.split[layer][node]
                NN.set_beta_config(layer, node, split, batch_id=batch_id)
    
    def splitNewNode(self):
        layer, node = self.next_candidate
        if layer in self.split and node in self.split[layer]:
            raise Exception(f"[Error] Candidate node {self.next_candidate} is already splited.")
        else:
            # print(f"Splitting new node: {self.next_candidate}")
            newConfigs = [SplitConfig(config = self) for _ in range(2)]
            newConfigs[0].add(layer, node, True)
            newConfigs[1].add(layer, node, False)
        return newConfigs

    def set_next_candidate(self, candidate):
        self.next_candidate = candidate