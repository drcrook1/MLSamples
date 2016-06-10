namespace MLSamples.FSharp.Regression
#load @"../../packages/FsLab.0.3.20/FsLab.fsx"
#load @"../Core.fsx"

module Logistic = 
    open MathNet.Numerics.LinearAlgebra
    open MLSamples.FSharp.Core

    let TwoClassPredict(theta:Vector<float>) (sample:Vector<float>) = 
        (theta * sample) |> sigmoid

    //sigmoid is for a specific theta, if you give it a training data set, you need to subtract column vector of y values
    //then square those two vectors.

    //one vs all prediction
    //returns a column matrix where each row represents the 
    let nClassPredict(allTheta:Matrix<float>) (trainingData:Matrix<float>) = 
        (trainingData * allTheta.Transpose())
        |> sigmoidM
        |> Matrix.toRowSeq
        |> Seq.map(fun r -> vector [r.AbsoluteMaximum()] )
        |> DenseMatrix.ofRowSeq
