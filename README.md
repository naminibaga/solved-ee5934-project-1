Download Link: https://assignmentchef.com/product/solved-ee5934-project-1
<br>
For this project, you will use a pre-trained deep neural network, <a href="https://arxiv.org/abs/1602.07360">SqueezeNet</a>, which is lightweight and runs fast on CPUs. Run the code below to load a pre-trained SqueezeNet from the PyTorch official model zoo.

In [2]:

# Test and set the device.if torch.cuda.is_available():    device = ‘cuda:0’else:    device = ‘cpu’print(‘Use’, device) # Download and load the pretrained SqueezeNet model.model = torchvision.models.squeezenet1_1(pretrained=True).to(device) # Disable the gradient computation with respect to model parameters.for param in model.parameters():    param.requires_grad = False

Use cuda:0

<h2>Data<a href="#Data">¶</a></h2>

For Task#1 and Task#2, use the images in folder Project1images where the filenames are the corresponding class labels. For example, 182.png is an image of Border Terrier, which is class 182 in ImageNet dataset. Please refer to <a href="https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a">this Gist snippet</a> for a complete list. The images are from ImageNet validation set, and so the pre-trained model has never “seen” them.

For Task#3, you may use the images in folder Project1style or any other images you like.

<h2>Helper Functions<a href="#Helper-Functions">¶</a></h2>

Most pre-trained models are trained on images that had been preprocessed by subtracting the per-color mean and dividing by the per-color standard deviation. Here are a few helper functions for performing and undoing this preprocessing.

In [3]:

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])IMAGENET_STD = np.array([0.229, 0.224, 0.225]) def preprocess(img, size=(224, 224)):    transform = T.Compose([        T.Resize(size),        T.ToTensor(),        T.Normalize(mean=IMAGENET_MEAN.tolist(),                    std=IMAGENET_STD.tolist()),        T.Lambda(lambda x: x[None]),    ])    return transform(img) def deprocess(img, should_rescale=True):    transform = T.Compose([        T.Lambda(lambda x: x[0]),        T.Normalize(mean=[0, 0, 0], std=(1.0 / IMAGENET_STD).tolist()),        T.Normalize(mean=(-IMAGENET_MEAN).tolist(), std=[1, 1, 1]),        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),        T.ToPILImage(),    ])    return transform(img) def rescale(x):    low, high = x.min(), x.max()    x_rescaled = (x – low) / (high – low)    return x_rescaled def blur_image(X, sigma=1):    X_np = X.cpu().clone().numpy()    X_np = gaussian_filter1d(X_np, sigma, axis=2)    X_np = gaussian_filter1d(X_np, sigma, axis=3)    X.copy_(torch.Tensor(X_np).type_as(X))    return X

<h1>Task#1 Adversarial Attack<a href="#Task#1-Adversarial-Attack">¶</a></h1>

The concept of “image gradients” can be used to study the stability of a network. Consider a state-of-the-art deep neural network that generalizes well on an object recognition task. We expect such network to be robust to small perturbations to its input, because small perturbations cannot change the object category of an image. However, it was shown in the following paper[1] that by applying an imperceptible non-random perturbation to a test image, it is possible to arbitrarily change the network’s prediction.

[1] <a href="https://arxiv.org/abs/1312.6199">Szegedy et al, “Intriguing properties of neural networks”, ICLR 2014</a>

Given an image and a target class, we can perform <strong>gradient ascent</strong> over the image to maximize the target class, stopping when the network classifies the image as the target class. While the perturbations seem negligible to humans, the network would classify the perturbed images wrongly.

Read the paper, and then implement the following function make_adversarial_attack to generate “fooling images”. For each image in Project1/images with class label $c$, generate a fooling image that will be classified into class $ c-1-d $ where $d$ is the last digit of your student number. Save each fooling image into the folder Project1/fooling_images with the filename {true_class}_{target_class}.png. You may confirm (optional) that the fooling image 182_9.png in the folder will be wrongly classified as ostrich (class 9 in ImageNet dataset).

For the image 182.png, show the difference map between the original image and the fooling image, and save it as 182_1x_diff.png. Magnify the difference by 10 times and save the resulting map as 182_10x_diff.png.

In [4]:

def make_adversarial_attack(X, target_y, model):    “””    Generate a fooling image that is close to X, but that the model classifies    as target_y.     Inputs:    – X: Input image; Tensor of shape (1, 3, 224, 224)    – target_y: An integer in the range [0, 1000)    – model: A pretrained CNN     Returns:    – X_fooling: An image that is close to X, but that is classifed as target_y    by the model.    “””        model.eval()        # Initialize our fooling image to the input image    X_fooling = X.clone().detach()    X_fooling.requires_grad = True     # you may change the learning rate and max_iter    learning_rate = 1     ##############################################################################    # TODO: Generate a fooling image X_fooling that the model will classify as   #    # the class target_y. You should perform gradient ascent on the score of the #    # target class, stopping when the model is fooled.                           #    # When computing an update step, first normalize the gradient:               #    #   dX = learning_rate * g / ||g||_2                                         #    ##############################################################################    loss = torch.nn.CrossEntropyLoss()    # The coefficients are tried out by human for several hours    # It is much better by using ones initialization ranther than randn initilization    r = 0.001*torch.ones(1,3,224,224,device=device)    r.requires_grad_(True)    # training the fooling image    for i in range(10000):        X_fooling = X_fooling + r        output = model(X_fooling)        output = torch.nn.functional.softmax(output,dim=1)        # IPNN loss function        cost =  0.00001*torch.norm(r,p=1) + loss(output,target_y)        # BP        cost.backward(retain_graph=True)        # Update the         with torch.no_grad():            r -= learning_rate*r.grad/torch.norm(r.grad,p=2)            # Manually zero the gradients after updating weights            r.grad.zero_()        # Predict y        predict_y = torch.argmax(output)        if predict_y == target_y:            print(‘Interations: %4d    Attack successed！’ %(i),’Predict_y:%d’%(predict_y),’norm(r,2):%f’%(torch.norm(r,p=2).item()))            break    ##############################################################################    #                             END OF YOUR CODE                               #    ##############################################################################    X_fooling = X_fooling.detach()        return X_fooling

In [5]:

############################################################################### TODO: 1. Compute the fooling images for the images under `Project1/images`.##       2. Show the 4 related images of the image ‘182.png’: original image, ##          fooling image, 182_1x_diff.png and2 182_10x_diff.png.              ################################################################################ read imagesimport glob,osfiles = glob.glob(‘images*.png’)# deal with imagesimages = []for file in files:    print(file)    input_image = PIL.Image.open(file)    input_tensor  = preprocess(input_image).cuda()    target_y = int(file[7:-4])-1-7 # A0206597U    print(target_y)    target_y = torch.tensor([target_y]).cuda()    output_tensor = make_adversarial_attack(input_tensor,target_y,model).cpu()    image = deprocess(output_tensor)    images.append(image)    # save the images    image.save(os.path.join(‘fooling_images’,file[7:-4]+’_’+str(target_y.item())+’.png’))  # plot imagesoriginal_182 = PIL.Image.open(files[1])fig1 = plt.figure()plt.imshow(original_182)plt.title(‘original_182’)fig2 = plt.figure()fooling_182 = images[1]plt.imshow(fooling_182)plt.title(‘fooling_182’)# diff imagestarget_y = int(files[1][7:-4])-1-7 # A0206597Utarget_y = torch.tensor([target_y]).cuda()input_tensor  = preprocess(original_182).cuda()output_tensor = make_adversarial_attack(input_tensor,target_y,model).cpu()difference_1x_182 = deprocess(input_tensor.cpu()-output_tensor,should_rescale=True)difference_10x_182 = deprocess((input_tensor.cpu()-output_tensor)*10,should_rescale=False)fig3 = plt.figure()plt.imshow(difference_1x_182)plt.title(‘difference_1x_182’)fig4 = plt.figure()plt.imshow(difference_10x_182)plt.title(‘difference_10x_182’)# save the diff imagesdifference_1x_182.save(‘182_1x_diff.png’)difference_10x_182.save(‘182_10x_diff.png’)###############################################################################                             END OF YOUR CODE                               ###############################################################################

images100.png

92

Interations:    7    Attack successed！ Predict_y:92 norm(r,2):2.178698

images182.png

174

Interations:    5    Attack successed！ Predict_y:174 norm(r,2):2.717995

images294.png

286

Interations:    4    Attack successed！ Predict_y:286 norm(r,2):2.086930

images366.png

358

Interations:    3    Attack successed！ Predict_y:358 norm(r,2):2.122715

images662.png

654

Interations:    5    Attack successed！ Predict_y:654 norm(r,2):2.321405

images85.png

77

Interations:   12    Attack successed！ Predict_y:77 norm(r,2):2.496330

Interations:    5    Attack successed！ Predict_y:174 norm(r,2):2.717996

<h1>Task#2 Class Visualization<a href="#Task#2-Class-Visualization">¶</a></h1>

By starting with a random noise image and performing gradient ascent on a target class, we can generate an image that the network will recognize as the target class. This idea was first presented in [2]; [3] extended this idea by suggesting several regularization techniques that can improve the quality of the generated image.

Concretely, let $I$ be an image and let $y$ be a target class. Let $s_y(I)$ be the score that a convolutional network assigns to the image $I$ for class $y$; note that these are raw unnormalized scores, not class probabilities. We wish to generate an image $I^*$ that achieves a high score for the class $y$ by solving the problem

$$ I^* = argmax_I (s_y(I) – R(I)) $$

where $R$ is a (possibly implicit) regularizer (note the sign of $R(I)$ in the argmax: we want to minimize this regularization term). We can solve this optimization problem using gradient ascent, computing gradients with respect to the generated image. We will use (explicit) L2 regularization of the form

$$ R(I) = lambda |I|_2^2 $$

<strong>and</strong> implicit regularization as suggested by [3] by periodically blurring the generated image. We can solve this problem using gradient ascent on the generated image.

In the cell below, complete the implementation of the create_class_visualization function.

[2] <a href="https://arxiv.org/abs/1312.6034">Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman, “Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps”, ICLR Workshop 2014</a>

[3] <a href="https://arxiv.org/abs/1506.06579">Yosinski et al, “Understanding Neural Networks Through Deep Visualization”, ICML 2015 Deep Learning Workshop</a>

In [6]:

def jitter(X, ox, oy):    “””    Helper function to randomly jitter an image.      Inputs    – X: PyTorch Tensor of shape (N, C, H, W)    – ox, oy: Integers giving number of pixels to jitter along W and H axes        Returns: A new PyTorch Tensor of shape (N, C, H, W)    “””    if ox != 0:        left = X[:, :, :, :-ox]        right = X[:, :, :, -ox:]        X = torch.cat([right, left], dim=3)    if oy != 0:        top = X[:, :, :-oy]        bottom = X[:, :, -oy:]        X = torch.cat([bottom, top], dim=2)    return X

In [7]:

def create_class_visualization(target_y, model, device, **kwargs):    ”’    Generate an image to maximize the score of target_y under a pretrained model.        Inputs:    – target_y: A list of two elements, where the first value is an integer in the range [0, 1000) giving the index of the                class, and the second value is the name of the class.    – model: A pretrained CNN that will be used to generate the image    – dtype: Torch datatype to use for computations        Keyword arguments:    – l2_reg: Strength of L2 regularization on the image    – learning_rate: How big of a step to take    – num_iterations: How many iterations to use    – blur_every: How often to blur the image as an implicit regularizer    – max_jitter: How much to gjitter the image as an implicit regularizer    – show_every: How often to show the intermediate result    ”’    model.to(device)    l2_reg = kwargs.pop(‘l2_reg’, 1e-3)    learning_rate = kwargs.pop(‘learning_rate’, 25)    num_iterations = kwargs.pop(‘num_iterations’, 100)    blur_every = kwargs.pop(‘blur_every’, 10)    max_jitter = kwargs.pop(‘max_jitter’, 16)    show_every = kwargs.pop(‘show_every’, 25)        # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.    img = torch.randn(1, 3, 224, 224).mul_(1.0).to(device).requires_grad_()     for t in range(num_iterations):        # Randomly jitter the image a bit; this gives slightly nicer results        ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)        img.data.copy_(jitter(img.data, ox, oy))         ########################################################################        # TODO: Use the model to compute the gradient of the score for the     #        # class target_y with respect to the pixels of the image, and make a   #        # gradient step on the image using the learning rate. Don’t forget the #        # L2 regularization term!                                              #        # Be very careful about the signs of elements in your code.            #        ########################################################################        scores = model(img)        loss = l2_reg*torch.norm(img,p=2)-2*scores[0,target_y[0]]        loss.backward()        # Update img        with torch.no_grad():            img -= learning_rate*img.grad            # Manually zero the gradients after updating weights            img.grad.zero_()        ########################################################################        #                             END OF YOUR CODE                         #        ########################################################################         # Undo the random jitter        img.data.copy_(jitter(img.data, -ox, -oy))         # As regularizer, clamp and periodically blur the image        for c in range(3):            lo = float(-IMAGENET_MEAN[c] / IMAGENET_STD[c])            hi = float((1.0 – IMAGENET_MEAN[c]) / IMAGENET_STD[c])            img.data[:, c].clamp_(min=lo, max=hi)        if t % blur_every == 0:            blur_image(img.data, sigma=0.5)         # Periodically show the image        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations – 1:            plt.imshow(deprocess(img.data.clone().cpu()))            class_name = target_y[1]            plt.title(‘%s
Iteration %d / %d’ % (class_name, t + 1, num_iterations))            plt.gcf().set_size_inches(4, 4)            plt.axis(‘off’)            plt.show()     return deprocess(img.data.cpu())

Once you have completed the implementation in the cell above, run the following cell to generate an image of a Tarantula:

In [8]:

target_y = [76, “Tarantula”]# target_y = 366 # Gorillaimport pdbout = create_class_visualization(target_y, model, device)

<h1>Task#3 Style Transfer<a href="#Task#3-Style-Transfer">¶</a></h1>

Another task which is closely related to image gradients is style transfer which has become a “cool” application in deep learning for computer vision applications. You need to study and implement the style transfer technique presented in the following paper [4] where the general idea is to take two images (a content image and a style image), and produce a new image that reflects the content of one but the artistic “style” of the other.

[4] <a href="http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf">Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. “Image style transfer using convolutional neural networks.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.</a>

Below is an example.

<h2>Compute the loss<a href="#Compute-the-loss">¶</a></h2>

To perform style transfer, you will need to first formulate a special loss function that matches the content and style of each respective image in the feature space, and then perform gradient descent on the pixels of the image itself.

The loss function contains two parts: <strong>content loss</strong> and <strong>style loss</strong>. Read the paper [4] for details about the losses and implement them below.

In [9]:

def content_loss(content_weight, content_current, content_original):    “””    Compute the content loss for style transfer.        Inputs:    – content_weight: Scalar giving the weighting for the content loss.    – content_current: features of the current image; this is a PyTorch Tensor of shape      (1, C_l, H_l, W_l).    – content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).        Returns:    – scalar content loss    “””        ##############################################################################    # TODO: Implement content loss function                                      #    # Note: It should not be very much code (less than 10 lines)                 #    ##############################################################################    loss = content_weight*torch.sum((content_current-content_original)**2)    return loss    ##############################################################################    #                             END OF YOUR CODE                               #    ##############################################################################  def gram_matrix(features):    “””    Compute the normalized Gram matrix from features.    The Gram matrix will be used to compute style loss.        Inputs:    – features: PyTorch Tensor of shape (N, C, H, W) giving features for      a batch of N images.        Returns:    – gram: PyTorch Tensor of shape (N, C, C) giving the      normalized Gram matrices for the N input images.    “””        ##############################################################################    # TODO: Implement the normalized Gram matrix compuation function             #    # Note: It should not be very much code (less than 10 lines)                 #    ##############################################################################    N,C,H,W = features.size()    features = torch.reshape(features,(N,C,-1))    gram = torch.bmm(features,features.permute(0,2,1))/float(C*H*W)    return gram    ##############################################################################    #                             END OF YOUR CODE                               #    ##############################################################################  def style_loss(feats, style_layers, style_targets, style_weights):    “””    Computes the style loss at a set of layers.        Inputs:    – feats: list of the features at every layer of the current image.    – style_layers: List of layer indices into feats giving the layers to include in the      style loss.    – style_targets: List of the same length as style_layers, where style_targets[i] is      a PyTorch Variable giving the Gram matrix of the source style image computed at      layer style_layers[i].    – style_weights: List of the same length as style_layers, where style_weights[i]      is a scalar giving the weight for the style loss at layer style_layers[i].          Returns:    – style_loss: A PyTorch Tensor holding a scalar giving the style loss.    “””        ##############################################################################    # TODO: Implement style loss function                                        #    # Note: It should not be very much code (less than 10 lines)                 #    ##############################################################################    loss = 0    for i,layer in enumerate(style_layers):        cur_gram = gram_matrix(feats[layer])        loss += style_weights[i]*torch.sum((cur_gram-style_targets[i])**2)/4    return loss    ##############################################################################    #                             END OF YOUR CODE                               #    ##############################################################################

<h2>Putting them together<a href="#Putting-them-together">¶</a></h2>

With these loss functions, you can now build your style transfer model. Implement the function below to perform style transfer. To test the model, you can use the content and style images that we have provided in Project1/style, or improvise using any image you like. Please save your output images in the Project1/style folder.

Design and carry out some experiments (on your own!) to analyse how <strong>the choice of layers</strong> and <strong>the weights</strong> will influence the output image. Write down your observations and analysis in the Markdown cell provided below.

In [146]:

def style_transfer(content_image, style_image, content_layer, content_weight,                   style_layers, style_weights, max_iter):    “””    Run style transfer!    You may first resize the image to a small size for fast computation.        Inputs:    – content_image: filename of content image    – style_image: filename of style image    – content_layer: an index indicating which layer to use for content loss    – content_weight: weighting on content loss    – style_layers: list of indices indicating which layers to use for style loss    – style_weights: list of weights to use for each layer in style_layers    – max_iter: max iterations of gradient updates        Returns:    – output_image: an image with content from the content_image and     style from the style image    “””    ##############################################################################    # TODO: Implement the function for style transfer.                           #    ##############################################################################    def extract_features(data,model):        feats = []        prev_feat = data        for module in model.features:                next_feat = module(prev_feat)                feats.append(next_feat)                prev_feat = next_feat        return feats    # remove white noise    def variance_loss(image):        loss = 0        # row        loss += torch.sum( ( image[:,:,1:,:] – image[:,:,:-1,:] )**2 )         # column        loss += torch.sum( ( image[:,:,:,1:] – image[:,:,:,:-1] )**2 )        return loss    # Content image    content_img = preprocess(PIL.Image.open(content_image)).to(device)    feats = extract_features(content_img,model)    content_target = feats[content_layer].clone()    # Style image    style_img = preprocess(PIL.Image.open(style_image)).to(device)    feats = extract_features(style_img,model)    style_target = []    for layer in style_layers:        style_target.append(gram_matrix(feats[layer].clone()))    # Transfer image    tran_img = content_img.clone().requires_grad_()    # Training    optimizer = torch.optim.Adam([tran_img],lr=1.0)    # Plot input images    f, axarr = plt.subplots(1,2)    axarr[0].axis(‘off’)    axarr[1].axis(‘off’)    axarr[0].set_title(‘Content Source Img.’)    axarr[1].set_title(‘Style Source Img.’)    axarr[0].imshow(deprocess(content_img.cpu()))    axarr[1].imshow(deprocess(style_img.cpu()))    plt.show()    plt.figure()    for i in range(max_iter):        # Enhancing contrast ratio        tran_img.data.clamp_(-1.5, 1.5)        # Init grad        optimizer.zero_grad()        feats = extract_features(tran_img, model)        # Compute loss        c_loss = content_loss(content_weight, feats[content_layer], content_target)        s_loss = style_loss(feats, style_layers, style_target, style_weights)        v_loss = variance_loss(tran_img)        total_loss = 0.2*c_loss + s_loss + 0.1*v_loss        # BP        total_loss.backward()        optimizer.step()    print(‘Iteration {}’.format(i))    plt.axis(‘off’)    plt.imshow(deprocess(tran_img.data.cpu()))    deprocess(tran_img.data.cpu()).save(‘style/engineering-the_scream.png’)    plt.show()    ##############################################################################    #                             END OF YOUR CODE                               #    ##############################################################################

In [147]:

############################################################################### TODO: 1. Choose one pair of images under ‘Project1/style’, and finish the  # #          neural style transfer task by calling the style_transfer function.##       2. Show the 3 related images: content image, style image and the     # #          generated style-transferred image.                                ################################################################################ Content layer comaparing params1 = {    ‘content_image’ : ‘style/engineering.jpg’,    ‘style_image’ : ‘style/the_scream.jpg’,    ‘content_layer’ : 3,    ‘content_weight’ : 0.1,     ‘style_layers’ : (1, 3, 4, 6, 7),    ‘style_weights’ : (20000, 10000,500, 12, 1),    ‘max_iter’ : 100}style_transfer(**params1)###############################################################################                             END OF YOUR CODE                               ###############################################################################

Iteration 99

As the content layer index increases, the transmitted image will look more like “style_image”. In addition, the ratio of $  alpha /  beta $ also controls the similarity with “content_image” or “style_image”. The larger the loss caused by the style layer, the more it looks like a style image.