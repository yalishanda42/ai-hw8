import XCTest
@testable import NeuralNetwork

final class NeuralNetworkTests: XCTestCase {
    func testNNInit0HiddenNodes() {
        let nn = NeuralNet(inputNodes: 2, hiddenNodes: 0, outputNodes: 1)
        XCTAssertEqual(nn.inputLayer.count, 2)
        XCTAssertEqual(nn.hiddenLayer.count, 0)
        XCTAssertEqual(nn.outputLayer.count, 1)
        XCTAssertEqual(nn.outputLayer[0].inputs.count, 2)
    }
    
    func testNNInit4HiddenNodes () {
        let nn = NeuralNet(inputNodes: 2, hiddenNodes: 4, outputNodes: 1)
        XCTAssertEqual(nn.inputLayer.count, 2)
        XCTAssertEqual(nn.hiddenLayer.count, 4)
        XCTAssertEqual(nn.hiddenLayer[0].inputs.count, 2)
        XCTAssertEqual(nn.outputLayer.count, 1)
        XCTAssertEqual(nn.outputLayer[0].inputs.count, 4)
    }

    static var allTests = [
        ("testNNInit0HiddenNodes", testNNInit0HiddenNodes),
        ("testNNInit4HiddenNodes", testNNInit4HiddenNodes),
    ]
}
