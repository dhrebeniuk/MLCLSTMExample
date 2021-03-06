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
        var tmpData = [Double]()
        for index in 0..<1000 {
            let a = Double(index) / 10.0
            let randomValue = (Double(arc4random()%100) / 100.0) / 20.0
            let value = pow(sin(a), 2.0) * (Double(index) * randomValue)
            tmpData.append(value)
        }
        
        
        generatedData.append(contentsOf: tmpData)
        
        lstmTrainEngine.setup()
        
        DispatchQueue.global().async {
            let floatData = self.generatedData.map { Float($0) }
            self.lstmTrainEngine.execTrainingLoop(trainingData: floatData) { _, epochIndex in
                if epochIndex%5 == 0 {
                    self.lstmTrainEngine.predictForecastGraph(evaluteData: floatData) { predictedData in
                        DispatchQueue.main.async {
                            self.epochIndex = epochIndex + 1
                            self.predictedData = predictedData.map { Double($0) }
                        }
                    }
                }
                else {
                    DispatchQueue.main.async {
                        self.epochIndex = epochIndex + 1
                    }
                }

            }
        }
    }
    
    
}
