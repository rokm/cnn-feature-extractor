root_dir = fileparts(mfilename('fullpath'));

% Root directory
addpath(root_dir);

% Caffe (binary)
addpath(fullfile(root_dir, 'external', 'caffe-bin', 'matlab'));