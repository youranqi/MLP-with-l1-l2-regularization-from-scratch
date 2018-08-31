function [s_z]=S(z,lambda)

if (abs(z)>lambda)
    s_z=sign(z)*(abs(z)-lambda);
else
    s_z=0;
end

end