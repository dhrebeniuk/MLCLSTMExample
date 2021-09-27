//
//  ContentView.swift
//  MLCLSTMExample
//
//  Created by Hrebeniuk Dmytro on 20.09.2021.
//

import SwiftUI
import SwiftUICharts

struct MainContentView: View {
    
    @EnvironmentObject var viewModel: MainContentViewModel

    
    var body: some View {
        VStack {
            VStack{
                LineView(data: self.viewModel.generatedData, title: "Generated Data", legend: nil)
                                
                LineView(data: self.viewModel.predictedData, title: "Predicted by Kalman Filter Data", legend: nil)

                Text("Epoch: \(viewModel.epochIndex)")

            }
        }.onAppear() {
            self.viewModel.setup()
        }.background(Color.black)
    }
}

struct MainContentView_Previews: PreviewProvider {
    static var previews: some View {
        MainContentView()
    }
}
