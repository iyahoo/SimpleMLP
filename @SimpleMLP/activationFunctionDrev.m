function y = activationFunctionDrev( x, selector )
    assert( isnumeric( x ), ...
            'x is type %s, not numeric', class( x ) );
    assert( ischar( selector ), ...
            'selector is type %s, not char', class( selector ) );

    switch lower( selector )
      case { 'sig', 'sigmoid' }
        y = x .* ( 1 - x );
      case { 'bip', 'bipolar' };
        y = 0.5 .* ( 1 + x ) .* ( 1 - x );
      case { 'relu' }
        if x >= 0
            y = 1;
        else
            y = 0;
        end
    end
end
