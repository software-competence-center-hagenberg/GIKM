function [distance] = distanceAutoencoder(y_data_new,AE)
[~,distance] = AutoencoderFiltering(y_data_new,AE);
return
