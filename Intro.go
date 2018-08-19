package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strings"
)

func main() {
	file, err := os.Open("Berkeley.csv")
	if err != nil {
		panic(err)
	}
	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	admissions := parseLines(lines)
	for _, admission := range admissions {
		fmt.Println(admission)
	}

}

func parseLines(lines [][]string) []admission {
	result := make([]admission, len(lines))
	for i, line := range lines {
		result[i] = admission{
			strings.TrimSpace(line[0]), //strings.TrimSpace() makes it robust against invalid csv
			strings.TrimSpace(line[1]),
			strings.TrimSpace(line[2]),
			strings.TrimSpace(line[3]),
		}
	}
	return result
}

type admission struct {
	Admit  string
	Gender string
	Dept   string
	Freq   string
}
