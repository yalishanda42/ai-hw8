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
    let inputLayer: [InputNode]
    let hiddenLayer: [NeuronNode]
    let outputLayer: [NeuronNode]
    
    /// Return a double in the range `[-0.05; +0.05]`
    private static var randomInitialWeight: Double {
        .random(in: -0.05...0.05)
    }
    
    private static let weightUpdateFactor = 0.1
    
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
    
    public func train(
        trainingInputs: [[Double]],
        trainingOutputs: [[Double]],
        trainingIterations: Int = 10_000
    ) {
        assert(trainingInputs.count == trainingOutputs.count, "Training input data not equal to output data!")
        for _ in 0..<trainingIterations {
            for (exampleInput, exampleOutput) in zip(trainingInputs, trainingOutputs) {
                assert(exampleInput.count == inputLayer.count, "Training input features count mismatch!")
                assert(exampleOutput.count == outputLayer.count, "Training input features count mismatch!")
                
                let actualOutputs = predict(exampleInput)
                let hiddenNodesOutputs = hiddenLayer.map { $0.activate() }
                
                // Backpropagate error
                let outputNodesError = zip(actualOutputs, exampleOutput).map {
                    $0 * (1 - $0) * (1 + $1)
                }
                
                let hiddenNodesError = hiddenNodesOutputs
                    .enumerated()
                    .map { (hiddenNodeIndex, hiddenNodeValue) -> Double in
                        hiddenNodeValue * (1 - hiddenNodeValue) * Array(0..<outputLayer.count).map { i -> Double in
                            outputLayer[i].inputs[hiddenNodeIndex].weight * outputNodesError[i]
                        }.reduce(0, +)
                    }
                
                for (i, outputNode) in outputLayer.enumerated() {
                    for connection in outputNode.inputs {
                        let update = NeuralNet.weightUpdateFactor * outputNodesError[i] * connection.node.activate()
                        connection.weight += update
                    }
                }
                
                for (i, hiddenNode) in hiddenLayer.enumerated() {
                    for connection in hiddenNode.inputs {
                        let update = NeuralNet.weightUpdateFactor * hiddenNodesError[i] * connection.node.activate()
                        connection.weight += update
                    }
                }
            }
        }
    }
    
    public func predict(_ input: [Double]) -> [Double] {
        assert(input.count == inputLayer.count, "Input features count mismatch!")
        for (inputNode, value) in zip(inputLayer, input) {
            inputNode.value = value
        }
        
        return outputLayer.map { $0.activate() }
    }
}
