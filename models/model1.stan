data {
  int<lower=0> n;  // number of data items
  int<lower=0> k;  // number of predictors
  matrix[n,k] X;   // predictor matrix
  vector[n] Y;          // outcome vector
}

parameters {
  real alpha;           // intercept
  matrix[k,1] beta;     // coefficients for predictors
  real<lower=0> sigma;  // error scale
}

transformed parameters{
  matrix[n,1] mu;
  vector[n] mu2;
  mu = X * beta + alpha; // regression
  mu2 = to_vector(mu);   // normal distribution 
}

model {
  // priors 
  for (i in 1:k) {            // normal priors for predictors
    beta[i] ~ normal(0, 100);
  }
  sigma ~ gamma(2, 0.1);      // gamma prior for standard deviation
  
  // likelihood
  Y ~ normal(mu2, sigma); 
}

generated quantities {
  vector[n] log_lik;
  for (i in 1:n)
    log_lik[i] = normal_lpdf(Y[i] | mu2[i], sigma);
}
