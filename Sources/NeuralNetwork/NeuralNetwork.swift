import RealModule  // for Double.exp(_:)

func sigmoid(_ x: Double) -> Double {
    return 1 / (1 + Double.exp(-x))
}

protocol Node: AnyObject {
    func activate() -> Double
}

class InputNode: Node {
    var value: Double
    
    init(value: Double = 0.0) {
        self.value = value
    }
    
    func activate() -> Double { value }
}

class Connection {
    let node: Node
    var weight: Double
    
    init(node: Node, weight: Double) {
        self.node = node
        self.weight = weight
    }
}

class NeuronNode: Node {
    let inputs: [Connection]
    
    init(inputs: [Connection]) {
        self.inputs = inputs
    }
    func activate() -> Double {
        let sum = inputs
            .map { $0.node.activate() * $0.weight }
            .reduce(0, +)
        return sigmoid(sum)
    }
}

/// Neural network containing 1 hidden layer that uses backpropagation.
public class NeuralNet {
    var inputLayer: [InputNode]
    let hiddenLayer: [NeuronNode]
    let outputLayer: [NeuronNode]
    
    /// Return a double in the range `[-0.05; +0.05]`
    private static var randomInitialWeight: Double {
        .random(in: -0.05...0.05)
    }
    
    public init(inputNodes: Int, hiddenNodes: Int, outputNodes: Int) {
        let inputLayerNodes = Array(repeating: InputNode(), count: inputNodes)
        self.inputLayer = inputLayerNodes
        let hiddenLayerNodes =  Array(0..<hiddenNodes).map { _ in
            NeuronNode(inputs: inputLayerNodes.map { inpnode in
                Connection(
                    node: inpnode,
                    weight: NeuralNet.randomInitialWeight
                )
            })
        }
        self.hiddenLayer = hiddenLayerNodes
        let outputLayerInputNodes: [Node] = hiddenLayerNodes.isEmpty
            ? inputLayerNodes
            : hiddenLayerNodes
        let outputLayerNodes = Array(0..<outputNodes).map { _ in
            NeuronNode(inputs: outputLayerInputNodes.map { inpnode in
                Connection(
                    node: inpnode,
                    weight: NeuralNet.randomInitialWeight
                )
            })
        }
        self.outputLayer = outputLayerNodes
    }
}
