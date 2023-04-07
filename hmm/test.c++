vector<int>* Model::viterbi(vector<int>* seq_obs)
{
    // longueur de la sequence d'observation
    int T = seq_obs->size();

    // Matrice pour stocker les probabilites max
    
    vector<vector<double>> delta(T, vector<double>(n, 0));

    // matrice pour stocker les etats precedents au fure et a mesure
    vector<vector<int>> phi(T, vector<int>(n, 0));

    // sequence d'etats caches plus probable
    vector<int>* q = new vector<int>(T, 0);

    // initialisation
    for (int i = 1; i <= n; i++) {
        delta[1][i] = pi[i] * B[i][(*seq_obs)[1]];
        phi[1][i] = 0;
    }

    // point 2 de l'algo

    for (int t = 2; t<= T; t++) {
        for(int j = 1; j<= n; j++) {
            double max_delta = 0.0;
            int max_i = 0;

            for(int i = 1; i<= n; i++) {
                double val = delta[t-1][i] * A[i][j] * B[j][(*seq_obs)[t]];
                if (val > max_delta){
                    max_delta = val;
                    max_i = i;
                }
            }

            delta[t][j] = max_delta;
            phi[t][j] = max_i;
        }
    }

    // point 3 et 4  de l'algo
    double max_proba = 0.0;
    int max_i = 0;
    for (int i = 1; i <= n; i++) {
        if (delta[T][i] > max_proba) {
            max_proba = delta[T][i];
            max_i = i;
        }
    }

    // point 5 de l'algo
    (*q)[T] = max_i;
    for (int t = T - 1; t >= 1; t--) {
        (*q)[t] = phi[t+1][(*q)[t+1]];
    }


    return q;

}