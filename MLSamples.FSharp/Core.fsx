namespace MLSamples.FSharp
#load @"../packages/FsLab.0.3.20/FsLab.fsx"

module Core = 
    open MathNet.Numerics.LinearAlgebra

    let square (x:Vector<float>) = x * x 
    let subtract (x:Vector<float>) (y:Vector<float>) = y - x
    let divideBy (x:float) (y:float) = y / x
    let divideVecBy (x:float) (y:Vector<float>) = y / x
    let multiply (x:float) (y:Vector<float>) = y * x

    //Sigmoid function
    //element wise division of a z matrix
    //1.0 ./ (1.0 + exp(-z));
    let private sigmoid (z:float) =
        1.0 / (1.0 + exp(-z))
    let private sigmoidM (z : Matrix<float>) : Matrix<float> =
        z.Map (fun x -> sigmoid(x))
    let private sigmoidV (z:Vector<float>) =
        z.Map(fun x -> sigmoid(x))
