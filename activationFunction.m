function y = activationFunction(x, selector)
    switch lower(selector)
      case {'sig', 'sigmoid'}
        y = 1 ./ (1 + exp(-x));
      case {'bip', 'bipolar'};
        y = -1.0 + 2.0 ./ (1 + exp(-x));
      case {'relu'}
        y = max(0,x);
    end
end
