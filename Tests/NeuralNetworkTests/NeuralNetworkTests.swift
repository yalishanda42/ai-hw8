import XCTest
@testable import NeuralNetwork

final class NeuralNetworkTests: XCTestCase {
    
    func testNeuronNodeActivationWhenInputSumIs0() {
        let n = NeuronNode(inputs: [
            .init(node: InputNode(value: +1), weight: 1),
            .init(node: InputNode(value: -1), weight: 1),
        ])
        let expected = 0.5
        
        let result = n.activate()
        
        XCTAssertEqual(result, expected)
    }
    
    func testNNInit0HiddenNodes() {
        let nn = NeuralNet(inputNodes: 2, hiddenNodes: 0, outputNodes: 1)
        XCTAssertEqual(nn.inputLayer.count, 2)
        XCTAssertEqual(nn.hiddenLayer.count, 0)
        XCTAssertEqual(nn.outputLayer.count, 1)
        XCTAssert(nn.outputLayer[0].inputs[0].node === nn.inputLayer[0])
        XCTAssert(nn.outputLayer[0].inputs[1].node === nn.inputLayer[1])
    }
    
    func testNNInitConnectionsCountCorrect() {
        let nn = NeuralNet(inputNodes: 2, hiddenNodes: 4, outputNodes: 1)
        XCTAssertEqual(nn.inputLayer.count, 2)
        XCTAssertEqual(nn.hiddenLayer.count, 4)
        XCTAssertEqual(nn.outputLayer.count, 1)
    }
    
    func testNNInitReferencesAndOrderAreCorrect() {
        let nn = NeuralNet(inputNodes: 2, hiddenNodes: 3, outputNodes: 2)
        XCTAssert(nn.hiddenLayer[0].inputs[0].node === nn.inputLayer[0])
        XCTAssert(nn.hiddenLayer[0].inputs[1].node === nn.inputLayer[1])
        XCTAssert(nn.hiddenLayer[1].inputs[0].node === nn.inputLayer[0])
        XCTAssert(nn.hiddenLayer[1].inputs[1].node === nn.inputLayer[1])
        XCTAssert(nn.hiddenLayer[2].inputs[0].node === nn.inputLayer[0])
        XCTAssert(nn.hiddenLayer[2].inputs[1].node === nn.inputLayer[1])
        XCTAssert(nn.outputLayer[0].inputs[0].node === nn.hiddenLayer[0])
        XCTAssert(nn.outputLayer[0].inputs[1].node === nn.hiddenLayer[1])
        XCTAssert(nn.outputLayer[0].inputs[2].node === nn.hiddenLayer[2])
        XCTAssert(nn.outputLayer[1].inputs[0].node === nn.hiddenLayer[0])
        XCTAssert(nn.outputLayer[1].inputs[1].node === nn.hiddenLayer[1])
        XCTAssert(nn.outputLayer[1].inputs[2].node === nn.hiddenLayer[2])
    }
    
    func testNNAND() {
        let trainingInput = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        let trainingOutput = [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
        ]
        let acceptableDeviationRate = 0.01
        
        let nn = NeuralNet(inputNodes: 2, hiddenNodes: 0, outputNodes: 1)
        
        measure {
            nn.train(trainingInputs: trainingInput, trainingOutputs: trainingOutput)
        }
        
        for (testInput, testOutput) in zip(trainingInput, trainingOutput) {
            let result = nn.predict(testInput)[0]
            print("Predicted \(testInput[0]) AND \(testInput[1]) = \(result)")
            let expectedMin = testOutput[0] * (1 - acceptableDeviationRate)
            let expectedMax = testOutput[0] * (1 + acceptableDeviationRate)
            XCTAssert(expectedMin < result && result < expectedMax)
        }
    }
    
    func testNNOR() {
        let trainingInput = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        let trainingOutput = [
            [0.0],
            [1.0],
            [1.0],
            [1.0],
        ]
        let acceptableDeviationRate = 0.01
        
        let nn = NeuralNet(inputNodes: 2, hiddenNodes: 0, outputNodes: 1)
        
        measure {
            nn.train(trainingInputs: trainingInput, trainingOutputs: trainingOutput)
        }
        
        for (testInput, testOutput) in zip(trainingInput, trainingOutput) {
            let result = nn.predict(testInput)[0]
            print("Predicted \(testInput[0]) OR \(testInput[1]) = \(result)")
            let expectedMin = testOutput[0] * (1 - acceptableDeviationRate)
            let expectedMax = testOutput[0] * (1 + acceptableDeviationRate)
            XCTAssert(expectedMin < result && result < expectedMax)
        }
    }
    
    func testNNXOR() {
        let trainingInput = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        let trainingOutput = [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ]
        let acceptableDeviationRate = 0.01
        
        let nn = NeuralNet(inputNodes: 2, hiddenNodes: 4, outputNodes: 1)
        
        measure {
            nn.train(trainingInputs: trainingInput, trainingOutputs: trainingOutput)
        }
        
        for (testInput, testOutput) in zip(trainingInput, trainingOutput) {
            let result = nn.predict(testInput)[0]
            print("Predicted \(testInput[0]) XOR \(testInput[1]) = \(result)")
            let expectedMin = testOutput[0] * (1 - acceptableDeviationRate)
            let expectedMax = testOutput[0] * (1 + acceptableDeviationRate)
            XCTAssert(expectedMin < result && result < expectedMax)
        }
    }
    

    static var allTests = [
        ("testNeuronNodeActivationWhenInputSumIs0", testNeuronNodeActivationWhenInputSumIs0),
        ("testNNInit0HiddenNodes", testNNInit0HiddenNodes),
        ("testNNInitConnectionsCountCorrect", testNNInitConnectionsCountCorrect),
        ("testNNInitReferencesAndOrderAreCorrect", testNNInitReferencesAndOrderAreCorrect),
        ("testNNAND", testNNAND),
        ("testNNOR", testNNOR),
        ("testNNXOR", testNNXOR),
    ]
}
