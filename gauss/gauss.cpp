#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <sstream>

using namespace std;

vector<double> multiplication(const vector<vector<double>> &A, const vector<double> &x){
    int n = A.size();
    vector<double> result(n, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            result[i] += A[i][j] * x[j];

    return result;
}

vector<double> gaussMethod(vector<vector<double>> A, vector<double> b){
    int n = A.size();
    for (int i = 0; i < n; i++){
        int rawWithMaxValue = i;
        double maxValue = abs(A[i][i]);
        #pragma omp parallel for reduction(max : maxValue)
        for (int p = i + 1; p < n; p++){
            double currentValue = abs(A[p][i]);
            if (currentValue > maxValue){
                maxValue = currentValue;
                rawWithMaxValue = p;
            }
        }

        if (rawWithMaxValue != i){
            swap(A[i], A[rawWithMaxValue]);
            swap(b[i], b[rawWithMaxValue]);
        }

        #pragma omp parallel for
        for (int p = i + 1; p < n; p++){
            double c = A[p][i]/A[i][i];
            for (int j = i; j < n; j++){
                A[p][j] -= c * A[i][j];
            }
            b[p] -= c * b[i];
        } // привели к верхнетреугольному виду
    }

    vector<double> x(n);
    for (int i = n-1; i >= 0; i--){
        x[i] = b[i] / A[i][i];
        #pragma omp parallel for
        for (int k = i - 1; k >= 0; k--)
            b[k] -= A[k][i] * x[i];
    }

    return x;
}

void setCoeffitients(vector<vector<double>> &A, vector<double> &b, int n){
    srand(time(0));
    A.resize(n, vector<double>(n));
    b.resize(n);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A[i][j] = rand() % 10 + 1; // чтобы не было нулевых строк
        }
        b[i] = rand() % 10;
    }
}

bool verification(const vector<vector<double>> &A, const vector<double> &x, const vector<double> &b, double threshold = 1e-6){
    vector<double> Ax = multiplication(A, x);
    for (int i = 0; i < b.size(); i++)
        if (abs(Ax[i] - b[i]) > threshold)
            return false;
    return true;
}

int main(){

    int n = 1000;
    vector<vector<double>> A;
    vector<double> b;
    double time_1;

    setCoeffitients(A, b, n);

    for (int threads = 1; threads <= 12; threads++){
        omp_set_num_threads(threads);
        double start_time = omp_get_wtime();
        vector<double> solution = gaussMethod(A, b);
        double end_time = omp_get_wtime();
        double delta_time = end_time - start_time;

        cout << "For " << threads << " threads solution is " << (verification(A, solution, b) ? "correct" : "incorrect") << endl;

        if (threads == 1)
            time_1 = delta_time;
        else
            cout << "Ускорение для " << threads << " процессов: " << time_1/delta_time << endl;
    }



    // это блок кода считывает из файла и выдает решение
    // ifstream fileA("A.txt");
    // ifstream fileB("b.txt");

    // vector<vector<double>> A;

    // string line;
    // while (getline(fileA, line)){
    //     istringstream iss(line);
    //     vector<double> row;
    //     double value;
    //     while (iss >> value)
    //         row.push_back(value);
    //     A.push_back(row);
    // }

    // int n = A.size();
    // vector<double> b(n);

    // for (int i = 0; i < n; i++)
    //     if (!(fileB >> b[i])){
    //         cerr << "b died" << endl;
    //         return 1;
    //     }
    
    // fileA.close();
    // fileB.close();

    // std::cout << "Матрица A:" << std::endl;
    // for (const auto& row : A) {
    //     for (const auto& elem : row) {
    //         std::cout << elem << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // // Вывод вектора b
    // std::cout << "Вектор b:" << std::endl;
    // for (const auto& elem : b) {
    //     std::cout << elem << " ";
    // }
    // std::cout << std::endl;

    // vector<double> solution = gaussMethod(A, b);

    // cout << "Решение:" << endl;
    // for (int i = 0; i < n; i++)
    //     cout << solution[i] << endl;

    return 0;
}