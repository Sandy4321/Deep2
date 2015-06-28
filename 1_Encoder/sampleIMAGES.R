sampleIMAGES <- function() {
    # sampleIMAGES
    # Returns 10000 patches for training
    
    #load IMAGES;    # load images from disk
    IMAGES <- readMat("IMAGES.mat")
    IMAGES <- IMAGES$IMAGES #512x512x10 matrix
    
    
    patchsize = 8;  # we'll use 8x8 patches 
    numpatches = 10000;
    
    # Initialize patches with zeros.  Your code will fill in this matrix--one
    # column per patch, 10000 columns. 
    # patches = zeros(patchsize*patchsize, numpatches);
    patches = matrix(0,patchsize*patchsize, numpatches);
    
    ## ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Fill in the variable called "patches" using data 
    #  from IMAGES.  
    #  
    #  IMAGES is a 3D array containing 10 images
    #  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
    #  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
    #  it. (The contrast on these images look a bit off because they have
    #  been preprocessed using using "whitening."  See the lecture notes for
    #  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
    #  patch corresponding to the pixels in the block (21,21) to (30,30) of
    #  Image 1
    
    set.seed(123)
    patches <- apply(patches, 2, function(x){
        nimage <- sample(1:10,1)
        nx <- sample(1:505,1)
        ny <- sample(1:505,1)
        img <- IMAGES[nx:(nx+patchsize-1), ny:(ny+patchsize-1), nimage]
        return(as.vector(img))
    })
    
    ## ---------------------------------------------------------------
    # For the autoencoder to work well we need to normalize the data
    # Specifically, since the output of the network is bounded between [0,1]
    # (due to the sigmoid activation function), we have to make sure 
    # the range of pixel values is also bounded between [0,1]
    return(normalizeData(patches));
}

## ---------------------------------------------------------------
normalizeData <- function(patches) {

    # Squash data to [0.1, 0.9] since we use sigmoid as the activation
    # function in the output layer
    
    # Remove DC (mean of images). 
    patches <- patches - mean(patches);
    
    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3 * sd(patches);
    
    patches <- apply(patches, c(1,2), function(x){
        max(min(x, pstd), -pstd) / pstd;
    })
    # Rescale from [-1,1] to [0.1,0.9]
    return((patches + 1) * 0.4 + 0.1);
}
