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

class BiasNode: Node {
    let bias: Double
    
    init(_ bias: Double) {
        self.bias = bias
    }
    
    func activate() -> Double { bias }
}

class NeuronNode: Node {
    let inputs: [Connection]
    let activationFunction: (Double) -> Double
    
    init(inputs: [Connection], activationFunction: @escaping (Double) -> Double = sigmoid(_:)) {
        self.inputs = inputs
        self.activationFunction = activationFunction
    }
    
    func activate() -> Double {
        let sum = inputs
            .map { $0.node.activate() * $0.weight }
            .reduce(0, +)
        return activationFunction(sum)
    }
}

/// Neural network containing 1 hidden layer that uses backpropagation.
public class NeuralNet {
    let inputLayer: [InputNode]
    let hiddenLayer: [NeuronNode]
    let outputLayer: [NeuronNode]
    
    let weightUpdateFactor: Double
    
    /// Return a double in the range `[-0.05; +0.05]`
    public class func deafultRandomInitialWeightFunction() -> Double {
        .random(in: -0.05...0.05)
    }
    
    public static let deafultWeightUpdateFactor = 0.1
    
    public init(
        inputNodes: Int,
        hiddenNodes: Int,
        outputNodes: Int,
        bias: Double? = nil,
        initialWeightFunction: () -> Double = NeuralNet.deafultRandomInitialWeightFunction,
        weightUpdateFactor: Double = NeuralNet.deafultWeightUpdateFactor
    ) {
        self.weightUpdateFactor = weightUpdateFactor
        let inputLayerNodes = Array(repeating: InputNode(), count: inputNodes)
        self.inputLayer = inputLayerNodes
        let hiddenLayerNodes = NeuralNet.initializeNeuronNodeLayer(nodesCount: hiddenNodes, inputLayer: inputLayerNodes, bias: bias, initialWeightFunction: initialWeightFunction)
        self.hiddenLayer = hiddenLayerNodes
        let outputLayerInputNodes: [Node] = hiddenLayerNodes.isEmpty
            ? inputLayerNodes
            : hiddenLayerNodes
        let outputLayerNodes = NeuralNet.initializeNeuronNodeLayer(nodesCount: outputNodes, inputLayer: outputLayerInputNodes, bias: bias, initialWeightFunction: initialWeightFunction)
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
                
                // Calculate error
                let outputNodesError = zip(actualOutputs, exampleOutput).map {
                    $0 * (1 - $0) * ($1 - $0)
                }
                
                let hiddenNodesError = hiddenNodesOutputs
                    .enumerated()
                    .map { (hiddenNodeIndex, hiddenNodeValue) -> Double in
                        hiddenNodeValue * (1 - hiddenNodeValue) * Array(0..<outputLayer.count).map { i -> Double in
                            outputLayer[i].inputs[hiddenNodeIndex].weight * outputNodesError[i]
                        }.reduce(0, +)
                    }
                
                // Backpropagate error
                updateLayerNodes(outputLayer, calculatedError: outputNodesError)
                updateLayerNodes(hiddenLayer, calculatedError: hiddenNodesError)
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

// MARK: - Helpers

private extension NeuralNet {
    class func initializeNeuronNodeLayer(
        nodesCount: Int,
        inputLayer: [Node],
        bias: Double?,
        initialWeightFunction: () -> Double
    ) -> [NeuronNode] {
        Array(0..<nodesCount).map { _ in
            NeuronNode(
                inputs: [Node](
                    inputLayer + [bias]
                                    .compactMap{ b in b }
                                    .compactMap { b in BiasNode(b)}
                    ).map { inputNode in
                        Connection(
                            node: inputNode,
                            weight: initialWeightFunction()
                        )
                    }
            )
        }
    }
    
    func updateLayerNodes(_ nodes: [NeuronNode], calculatedError: [Double]) {
        assert(nodes.count == calculatedError.count, "Not all nodes have errors calculated!")
        for (i, outputNode) in nodes.enumerated() {
            for connection in outputNode.inputs {
                let update = weightUpdateFactor * calculatedError[i] * connection.node.activate()
                connection.weight += update
            }
        }
    }
}
