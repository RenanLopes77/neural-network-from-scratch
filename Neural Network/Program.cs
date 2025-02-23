var neuralNetwork = CreateNeuralNetwork();


NeuralNetwork CreateNeuralNetwork()
{
    return new NeuralNetwork();
}

public class NeuralNetwork
{
    public List<Layer> Layers { get; private set; }
    public double Prediction { get; private set; }
    public ILossFuntion Loss { get; private set; }
}

public class Layer
{
}

public interface ILossFuntion
{
}