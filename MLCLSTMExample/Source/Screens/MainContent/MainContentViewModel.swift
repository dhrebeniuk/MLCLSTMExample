//
//  MainContentViewModel.swift
//  MLCLSTMExample
//
//  Created by Hrebeniuk Dmytro on 20.09.2021.
//

import Foundation
import Combine
import SwiftUI


class MainContentViewModel: ObservableObject {
    
    @Published var generatedData: [Double] = [Double]()
    @Published var predictedData: [Double] = [Double]()
    @Published var epochIndex: Int = 0

    let lstmTrainEngine = LSTMTrainEngine()
    
    
    
    func setup () {
        let url = Bundle.main.url(forResource: "1", withExtension: "csv")
        let content = url.flatMap { try? String(contentsOf: $0) }
        var lines = content?.split(separator: "\n")
        lines?.removeFirst()
        let actionsData = lines.map { $0.map { $0.split(separator: ",")[0..<3].map { String($0) } } }
        let actionsValues = actionsData.map { $0.map { $0.map { ($0 as NSString).doubleValue } } }
        let tmpData = Array((actionsValues.flatMap { $0.map { $0[0] } } ?? [Double]())[0..<1000])
        
//        var tmpData = [Double]()
//        for index in 0..<1000 {
//            let a = Double(index) / 10.0
//            let randomValue = (Double(arc4random()%100) / 100.0) / 20.0
//            let value = pow(sin(a), 2.0) * (Double(index) * randomValue)
//            tmpData.append(value)
//        }
        


        
        
        generatedData.append(contentsOf: tmpData)
        
        
        self.generatedData = generatedData

        let kalmanFilter = KalmanFilter<Double>(varianceProcess: 0.01)

        let filteredData = kalmanFilter.fit(values: tmpData)
        
        self.predictedData = filteredData
        

        //0.6465630896462902, 0.7733197394140312, 0.14656308964629017

        lstmTrainEngine.setup()
        
//        DispatchQueue.global().async {
//            let floatData = self.generatedData.map { Float($0) }
//            self.lstmTrainEngine.execTrainingLoop(trainingData: floatData) { _, epochIndex in
//                if epochIndex%5 == 0 {
//                    self.lstmTrainEngine.predictForecastGraph(evaluteData: floatData) { predictedData in
//                        DispatchQueue.main.async {
//                            self.epochIndex = epochIndex + 1
//                            self.predictedData = predictedData.map { Double($0) }
//                        }
//                    }
//                }
//                else {
//                    DispatchQueue.main.async {
//                        self.epochIndex = epochIndex + 1
//                    }
//                }
//
//            }
//        }
    }
    
    
}
