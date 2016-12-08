classdef CnnFeatureExtractor < handle
    % CNNFEATUREEXTRACTOR - Caffe based CNN feature extractor
    %
    % This class provides a wrapper for MatCaffe to simplify the use of its
    % convolutional neural network models as feature extractors.
    %
    % The functionality is loosely based on that of Ross Girschick's
    % original R-CNN, but aims to be more flexible and accomodates the
    % modern MatCaffe API.
    %
    % (C) 2016, Rok Mandeljc

    properties
        % CNN model that we are using
        cnn

        % Name of layer to extract features from (default to use if not
        % provided when calling EXTRACT() method)
        layer_name

        % Pixel means; if non-empty, the values are subtracted from the
        % image before it is feed to CNN
        pixel_means

        % Pixel scale; if non-empty, the image is scaled with this value
        % before it is fed to CNN
        pixel_scale

        % Original dimensions data blob dimensions
        % [ width, height, channels, batch_size ]
        input_dim

        % Desired batch size; overrides the one specified by the network
        % model
        batch_size

        % Cropping parameters
        square_crop
        padding
    end

    methods
        function self = CnnFeatureExtractor (proto_file, data_file, varargin)
            % self = CNNFEATUREEXTRACTOR (proto_file, data_file, varargin)
            %
            % Creates a CnnFeatureExtractor.
            %
            % Input:
            %  - proto_file: network model definition (.proto file)
            %  - data_file: network weights
            %  - varargin: key/value pairs, specifying additional options:
            %    - layer_name: default layer name to use for feature
            %      extraction (default: '', meaning that the name needs to
            %      be provided at extraction time)
            %    - pixel_means: pixel mean values that are subtracted from
            %      the image before it is fed to the network. Can be either
            %      a filename pointing to the binary file in Caffe's
            %      format, or numeric - either a matrix with dimensions
            %      matching those of the input data blob, or a 3-D vector
            %      that specifies the mean value for each image channel
            %      (default: [], meaning no mean subtraction)
            %    - pixel_scale: pixel scale factor; applied to image before
            %      passing it to the network (default: [], meaning no
            %      scaling)
            %    - batch_size: batch size (default: 256)
            %    - square_crop: whether to perform square cropping instead
            %      of warp cropping (default: false)
            %    - padding: additional padding to apply around crops
            %      (default: 0)
            %    - use_gpu: whether to use GPU or not. Note that this
            %      setting is Caffe-wide, so it applies to all loaded
            %      networks (default: false)
            %
            % Output:
            %  - self: a @CnnFeatureExtractor instance

            % Input options
            parser = inputParser();
            parser.addParameter('layer_name', '', @ischar);
            parser.addParameter('pixel_means', []);
            parser.addParameter('pixel_scale', []);
            parser.addParameter('batch_size', 256, @isscalar);
            parser.addParameter('square_crop', false, @islogical);
            parser.addParameter('padding', 0, @isnumeric);
            parser.addParameter('use_gpu', false, @islogical);
            parser.parse(varargin{:});

            self.batch_size = parser.Results.batch_size;
            self.square_crop = parser.Results.square_crop;
            self.padding = parser.Results.padding;

            if parser.Results.use_gpu,
                caffe.set_mode_gpu();
            else
                caffe.set_mode_cpu();
            end

            %% Pixel means
            pixel_means = parser.Results.pixel_means;
            if ~isempty(pixel_means),
                if ischar(pixel_means),
                    % Filename
                    [ ~, ~, ext ] = fileparts(pixel_means);
                    if isequal(ext, '.mat'),
                        % MAT file, containing 'pixel_means' variable
                        tmp = load(pixel_means);
                        self.pixel_means = tmp.pixel_means;
                    else
                        % Caffe binary format
                        self.pixel_means = caffe.io.read_mean(pixel_means);
                    end
                elseif isnumeric(pixel_means),
                    % Numeric values
                    self.pixel_means = squeeze(pixel_means);
                end
            end

            %% Pixel scale factor
            self.pixel_scale = parser.Results.pixel_scale;

            %% Default layer name
            self.layer_name = parser.Results.layer_name;

            %% Load network
            assert(exist(proto_file, 'file') ~= 0, 'Invalid network proto file!');
            assert(exist(data_file, 'file') ~= 0, 'Invalid network data file!');

            self.cnn = caffe.get_net(proto_file, data_file , 'test');

            % Make sure that default layer name is available
            if ~isempty(self.layer_name),
                assert(ismember(self.layer_name, self.cnn.blob_names), 'Invalid output layer/blob name!');
            end

            % Get input blob dimensions
            self.input_dim = self.cnn.blobs('data').shape;
        end

        function [ output, time ] = extract (self, I, varargin)
            % [ output, time ] = EXTRACT (self, I, varargin)
            %
            % Extracts CNN features from specified regions in the input
            % image, as response values in the specified layer.
            %
            % Input:
            %  - self: @CnnFeatureExtractor instance
            %  - I: input image (HxWx3 or HxWx1)
            %  - varargin: additional options:
            %    - regions: 4xN matrix defining the regions in
            %      [ x1, y1, x2, y2 ]' format. If not provided, a single
            %      region encompassing the whole image is used instead.
            %    - layer_name: name of the layer to extract features from.
            %      If not provided, the layer name, provided at extractor's
            %      construction time is used. If neither are provided, an
            %      error is raised.
            %
            % Output:
            %  - output: DxN matrix of responses, where D is response
            %    dimension and N is number of regions
            %  - time: time required to obtain the output

            parser = inputParser();
            parser.addParameter('regions', [], @isnumeric);
            parser.addParameter('layer_name', '', @ischar);
            parser.parse(varargin{:});

            t = tic();

            % Default box: whole image
            regions = parser.Results.regions;
            if isempty(regions),
                regions = [ 1, 1, size(I,2), size(I,1) ]';
            end
            assert(size(regions,1) == 4, 'boxes must be 4xN matrix!');
            num_regions = size(regions, 2);

            % Layer
            layer_name = parser.Results.layer_name;
            if isempty(layer_name),
                layer_name = self.layer_name;
            else
                assert(ismember(layer_name, self.cnn.blob_names), 'Invalid layer name!');
            end
            assert(~isempty(layer_name), 'No layer name was specified (and no default was given at construction)!');

            output_blob = self.cnn.blobs(layer_name);
            output_dim = output_blob.shape();
            output_dim = output_dim(1);

            % Allocate output
            output = nan(output_dim(1), num_regions);

            %% Prepare input image
            if self.input_dim(3) == 3,
                % Convert image to single-precision, BGR
                if size(I, 3) == 1,
                    I = repmat(I, 1, 1, 3);
                end
                Ic = single(I(:, :, [ 3, 2, 1]));
            elseif self.input_dim(3) == 1,
                % Convert image to single-precision, grayscale
                if size(I, 3) == 3,
                    I = rgb2gray(I);
                end
                Ic = single(I);
            else
                error('Invalid input image format!');
            end

            %% Process all batches
            batch_size = self.batch_size;
            num_batches = ceil(num_regions / batch_size);

            idx = 1;
            for b = 1:num_batches,
                cur_batch_size = min(num_regions, batch_size);

                fprintf('Batch #%d: %d regions (%d ~ %d)\n', b, cur_batch_size, idx, idx+cur_batch_size-1);

                % Process a batch
                output(:,idx:idx+cur_batch_size-1) = self.extract_batch(Ic, regions(:,idx:idx+cur_batch_size-1), layer_name);

                idx = idx + cur_batch_size;
                num_regions = num_regions - cur_batch_size;
            end

            assert(num_regions == 0, 'Bug in code!');

            % Record the time
            time = toc(t);
        end
    end

    methods (Access = private)
        function output = extract_batch (self, I, regions, layer_name)
            % output = EXTRACT_BATCH (self, I, regions, layer_name)
            %
            % Extracts batch of CNN features from specified regions in the
            % input image, as response values in the specified layer.

            num_regions = size(regions, 2);

            %% Prepare data
            % Allocate data
            data_size = [ self.input_dim(1:3), num_regions ];
            data = zeros(data_size, 'single');

            % Crop all regions
            for r = 1:num_regions,
                Ic = crop_region(self, I, regions(:,r));

                % Switch dimensions to achieve Caffee-compatible layout
                data(:,:,:,r) = permute(Ic, [ 2, 1, 3 ]);
            end

            %% Push data through CNN
            data_blob = self.cnn.blobs('data');

            data_blob.reshape(data_size);
            data_blob.set_data(data);

            % Do a forward pass
            self.cnn.forward_prefilled();

            %% Fetch data from specified layer
            output = self.cnn.blobs(layer_name).get_data();
        end

        function window = crop_region (self, I, region)
            % window = CROP_REGION (self, I, region)
            %
            % Crops the specified region out of the input image.

            % Crop size; same as the size of the input blob in our CNN
            crop_size = self.input_dim(1);
            square_crop = self.square_crop;
            padding = self.padding;

            %% Determine pad size
            pad_w = 0;
            pad_h = 0;
            crop_width = crop_size;
            crop_height = crop_size;

            if padding > 0 || square_crop,
                scale = crop_size/(crop_size - padding*2);

                half_height = (region(4) - region(2) + 1)/2;
                half_width = (region(3) - region(1) + 1)/2;
                center = [ region(1)+half_width, region(2)+half_height ];

                if square_crop,
                    % Make the box a tight square
                    if half_height > half_width,
                        half_width = half_height;
                    else
                        half_height = half_width;
                    end
                end
                region = round([ center, center] + scale*[-half_width, -half_height, half_width, half_height]);

                unclipped_height = region(4) - region(2) + 1;
                unclipped_width = region(3) - region(1) + 1;

                pad_x1 = max(0, 1 - region(1));
                pad_y1 = max(0, 1 - region(2));

                % clipped bbox
                region(1) = max(1, region(1));
                region(2) = max(1, region(2));
                region(3) = min(size(I,2), region(3));
                region(4) = min(size(I,1), region(4));

                clipped_height = region(4) - region(2) + 1;
                clipped_width = region(3) - region(1) + 1;
                scale_x = crop_size / unclipped_width;
                scale_y = crop_size / unclipped_height;

                crop_width = round(clipped_width*scale_x);
                crop_height = round(clipped_height*scale_y);

                pad_x1 = round(pad_x1*scale_x);
                pad_y1 = round(pad_y1*scale_y);
                pad_h = pad_y1;
                pad_w = pad_x1;

                if pad_y1 + crop_height > crop_size,
                    crop_height = crop_size - pad_y1;
                end
                if pad_x1 + crop_width > crop_size,
                    crop_width = crop_size - pad_x1;
                end
            end

            % Cut the region from input image
            x1 = min(max( round(region(1)), 1), size(I, 2));
            x2 = min(max( round(region(3)), 1), size(I, 2));
            y1 = min(max( round(region(2)), 1), size(I, 1));
            y2 = min(max( round(region(4)), 1), size(I, 1));

            window = I(y1:y2, x1:x2, :);

            % Turn off antialiasing to better match OpenCV's bilinear
            % interpolation in Caffe's WindowDataLayer.
            tmp = imresize(window, [ crop_height, crop_width ], 'bilinear', 'antialiasing', false);

            % Subtract pixel means
            if ~isempty(self.pixel_means),
                if isvector(self.pixel_means),
                    pixel_means = reshape(self.pixel_means, 1, 1, 3);
                    tmp = bsxfun(@minus, tmp, pixel_means);
                else
                    tmp = tmp - self.pixel_means(pad_h+(1:crop_height), pad_w+(1:crop_width), :);
                end
            end

            % Scale
            if ~isempty(self.pixel_scale),
                tmp = bsxfun(@times, tmp, self.pixel_scale);
            end

            window = zeros(crop_size, crop_size, self.input_dim(3), 'single');
            window(pad_h+(1:crop_height), pad_w+(1:crop_width), :) = tmp;
        end
    end
end
