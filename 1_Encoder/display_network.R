display_network <- function(X) {
    L <- nrow(X)
    M <- ncol(X)

    sz=sqrt(L);
    buf=1;
    
    if (floor(sqrt(M))^2 != M) {
        n=ceiling(sqrt(M));
        while (M%%n!=0 & n<1.2*sqrt(M)) {n=n+1}
        m=ceiling(M/n);
    } else {
        n=sqrt(M);
        m=n;
    }
    
    array=matrix(-1,buf+m*(sz+buf),buf+n*(sz+buf));
    
    X <- X-mean(X)
    
    k=1;
    for (i in 1:m) {
        for (j in 1:n) {
            if (k>M) {break} 
            clim <- max(abs(X[,k]));
            array[buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz)]=matrix(X[,k],sz,sz)/clim;
            k=k+1;
        }
    }        
    
    par(xaxt="n", yaxt="n")
    image(array, col=heat.colors(16))
}