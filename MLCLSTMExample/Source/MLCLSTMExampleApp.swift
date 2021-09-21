//
//  MLCLSTMExampleApp.swift
//  MLCLSTMExample
//
//  Created by Hrebeniuk Dmytro on 20.09.2021.
//

import SwiftUI

@main
struct MLCLSTMExampleApp: App {
    var body: some Scene {
        WindowGroup {
            MainContentView().environmentObject(MainContentViewModel())
        }
    }
}
