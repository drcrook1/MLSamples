#load @"../packages/FsLab.0.3.20/FsLab.fsx"
#load @"./Core.fsx"
#load @"./Regression/Linear.fsx"

open MathNet.Numerics.LinearAlgebra
open MLSamples.FSharp.Regression.Linear

//Create some fake training Data
let tData = matrix [[ 1.0; 2.0; 5.0]
                    [1.0; 3.0; 6.0]
                    [1.0; 3.0; 7.0]]  
//Create a few fake actuals             
let yVals = vector [ 5.0; 6.0; 7.0]  
//provide a starting point for theta    
let theta = vector [0.0; 0.0; 0.0]    
//Whats our initial error?
tData |> MSSE theta yVals
//Create Linear Configuration
let config = { MaxIterations = 1000; MinDelta = 0.01; Alpha = 0.01}
//Train model and return weights
let weights = tData |> TrainModel yVals config
//Whats our error after trained model?
tData |> MSSE weights yVals
