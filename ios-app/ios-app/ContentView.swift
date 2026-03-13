//
//  ContentView.swift
//  ios-app
//
//  Created by Ridhwik Kalgaonkar on 10/25/24.
//

import SwiftUI

struct ContentView: View {
    
    @State private var recipeName: String = ""
    let recipeClient = RecipeClient()
    @State private var recipes: [Recipe] = []
    
    var body: some View {
        VStack {
            TextField("Enter name", text: $recipeName).textFieldStyle(.roundedBorder).onSubmit {
                Task {
                    do {
                        recipes = try await recipeClient.searchRecipe(recipeName)
                    } catch {
                        print("Error: \(error)")
                    }
                }
            }.padding()
            
            List(recipes) { recipe in
                HStack {
                    AsyncImage(url: recipe.featuredImage, content: {image in image.resizable().clipShape(RoundedRectangle(cornerRadius: 25.0, style: .continuous)).frame(width: 100, height: 100)}, placeholder: {ProgressView("Loading...")})
                    Text(recipe.title)
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
