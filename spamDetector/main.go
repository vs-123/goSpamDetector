// By Vahin Sharma
package main

import (
  "fmt"
  cv "spamDetector/countVectorizer"
  deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
  "math"
  "os"
  "bufio"
)

func main() {
  corpus := []string{
    "thats so cool",
    "hey how are you",
    "wassup",
    "damn thats cool",
    "hey how are you doing",
    "this is totally not a spam",
    
    "important you could be entitled up to",
    "join this telegram to earn money",
    "join this telegram",
    "hey there do you want free money",
    "join this for some money",
    "hey would you like some free money",
    "You've Won! Winning an unexpected prize sounds great in theory",
    "Verify Your Bank Account",
  }
  y := []float64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}
  vectorizer := cv.Fit(corpus)
  data := training.Examples{}
  for index, i := range corpus {
    data = append(data, training.Example{vectorizer.Transform(i), []float64{y[index]}})
  }
  
  inputSize := len(data[0].Input)
  
  n := deep.NewNeural(&deep.Config{
    Inputs: inputSize,
    Layout: []int{(inputSize+1)*2, 1},
    Activation: deep.ActivationTanh,
    Mode: deep.ModeMultiLabel,
    Weight: deep.NewNormal(1, 0),
    Loss: deep.LossMeanSquared,
    Bias: true,
  })
  opt := training.NewSGD(0.05, 0.1, 1e-6, true)
  trainer := training.NewTrainer(opt, 0)
  trainingData, validationData := data.Split(0.5)
  trainer.Train(n, trainingData, validationData, 1000)
  
  scanner := bufio.NewScanner(os.Stdin)
  
  for {
    fmt.Print("Enter text to detect spam or not: ")
    scanner.Scan()
    transformedText := vectorizer.Transform(scanner.Text())
    prediction := n.Predict(transformedText)[0]
    
    fmt.Println()
    
    if math.Round(prediction) == 1 {
      fmt.Println("It's a spam")
    } else {
      fmt.Println("It's not a spam")
    }
    
    fmt.Println()
  }  
}