function VOCopts = get_kitti_opts(path)

tmp = pwd;
cd(path);
try
  addpath('KITTIcode');
  KITTIinit;
catch
  rmpath('KITTIcode');
  cd(tmp);
  error(sprintf('KITTIcode directory not found under %s', path));
end
rmpath('KITTIcode');
cd(tmp);
