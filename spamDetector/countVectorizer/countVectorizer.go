package countVectorizer

import (
  "strings"
  "regexp"
)

type CountVector struct {
  dictionary []string
}

func cleanString(x string) string {
  re := regexp.MustCompile(`(?m)([^a-zA-Z0-9 ])`)
  return re.ReplaceAllString(strings.ToLower(x), "")
}

func contains(x []string, y string) bool {
  for _, i := range x {
    if i == y {
      return true
    }
  }
  return false
}

func Fit(x []string) CountVector {
  dict := []string{}
  for _, text := range x {
    for _, word := range strings.Split(cleanString(text), " ") {
      if !contains(dict, word) {
        dict = append(dict, word)
      }
    }
  }
  return CountVector{dictionary: dict}
}

func (vectorizer CountVector) Transform(text string) []float64 {
  text = cleanString(text)
  
  count := make(map[string]float64)
  var out []float64
  
  for _, i := range vectorizer.dictionary {
    count[i] = 0
  }
  
  for _, i := range strings.Split(text, " ") {
    if _, ok := count[i]; ok {
      count[i]++
    }
  }
  
  for _, i := range vectorizer.dictionary {
    out = append(out, count[i])
  }
  
  return out
}