function [] = extractImageFeatures(imageFolder,featureType)
imds = imageDatastore(imageFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
switch lower(featureType)
    case 'alexnet'
        net = alexnet;
        imds.ReadFcn = @(loc)preprocess_alexnet(loc);
        parfor i = 1:numel(imds.Files)
            fprintf('Exracting alexnet features of %s\n',imds.Files{i});
            alexnet_features = activations(net,readimage(imds,i),'fc6','OutputAs','columns');
            [~,~,c] = fileparts(imds.Files{i});
            parsave_alexnet(strrep(imds.Files{i},c,'_alexnet.mat'),alexnet_features);
        end
    case 'vgg16'
        net = vgg16();
        imds.ReadFcn = @(loc)preprocess_vgg16(loc);
        parfor i = 1:numel(imds.Files)
            fprintf('Exracting vgg16 features of %s\n',imds.Files{i});
            vgg16_features = activations(net,readimage(imds,i),'fc6','OutputAs','columns');
            [~,~,c] = fileparts(imds.Files{i});
            parsave_vgg16(strrep(imds.Files{i},c,'_vgg16.mat'),vgg16_features);
        end
    case 'resnet50'
        net = resnet50;
        imds.ReadFcn = @(loc)preprocess_resnet50(loc);
        parfor i = 1:numel(imds.Files)
            fprintf('Exracting resnet50 features of %s\n',imds.Files{i});
            resnet50_features = squeeze(activations(net,readimage(imds,i),'avg_pool'));
            [~,~,c] = fileparts(imds.Files{i});
            parsave_resnet50(strrep(imds.Files{i},c,'_resnet50.mat'),resnet50_features);
        end
    case 'resnet18'
        net = resnet18;
        imds.ReadFcn = @(loc)preprocess_resnet18(loc);
        parfor i = 1:numel(imds.Files)
            fprintf('Exracting resnet18 features of %s\n',imds.Files{i});
            resnet18_features = activations(net,readimage(imds,i),'pool5','OutputAs','columns');
            [~,~,c] = fileparts(imds.Files{i});
            parsave_resnet18(strrep(imds.Files{i},c,'_resnet18.mat'),resnet18_features);
        end
end
fprintf('Features extracted.\n');
return


function [] = parsave_alexnet(fname,alexnet_features)
save(fname,'alexnet_features');
return

function [] = parsave_vgg16(fname,vgg16_features)
save(fname,'vgg16_features');
return

function [] = parsave_resnet50(fname,resnet50_features)
save(fname,'resnet50_features');
return

function [] = parsave_resnet18(fname,resnet18_features)
save(fname,'resnet18_features');
return

function [I] = preprocess_alexnet(loc)
I = imread(loc);
if ismatrix(I)
    I = cat(3,I,I,I);
end
I = imresize(I,[227,227]);
return

function [I] = preprocess_vgg16(loc)
I = imread(loc);
if ismatrix(I)
    I = cat(3,I,I,I);
end
I = imresize(I,[224,224]);
return

function [I] = preprocess_resnet50(loc)
I = imread(loc);
if ismatrix(I)
    I = cat(3,I,I,I);
end
I = imresize(I,[224,224]);
return

function [I] = preprocess_resnet18(loc)
I = imread(loc);
if ismatrix(I)
    I = cat(3,I,I,I);
end
I = imresize(I,[224,224]);
return






