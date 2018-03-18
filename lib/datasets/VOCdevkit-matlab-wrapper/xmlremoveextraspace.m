close all
clear
clc

folder_path = '/home/zhefanye/Documents/Programs/code/object_detection/faster-rcnn.pytorch/data/progress/Annotations/';
anno_dir = dir([folder_path '*.xml']);

for i = 1:numel(anno_dir)
    filePath = [folder_path anno_dir(i).name];
    % Read file.
    fId = fopen(filePath, 'r');
    fileContents = fread(fId, '*char')';
    fclose(fId);
    % Write new file.
    fId = fopen(filePath, 'w');
    % Remove extra lines.
    haha = '<?xml version="1.0" encoding="utf-8"?>';
    output = regexprep(fileContents, '   ', '\t');
    fwrite(fId, output(length(haha)+2:end));
    fclose(fId);
end
