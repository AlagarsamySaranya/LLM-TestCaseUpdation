public static double uniformCdf(int k, int n) {
        if(k<0 || n<1) {
            throw new IllegalArgumentException("All the parameters must be positive and n larger than 1.");
        }
        k = Math.min(k, n);
        
        double probabilitySum = k*uniform(n);
        
        return probabilitySum;
    }