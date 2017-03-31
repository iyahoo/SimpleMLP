function y = activationFunction( x, selector )
    assert( isnumeric( x ), ...
            'x is type %s, not numeric', class( x ) );
    assert( ischar( selector ), ...
            'selector is type %s, not char', class( selector ) );

    switch lower( selector )
      case { 'sig', 'sigmoid' }
        y = 1 ./ ( 1 + exp( -x ));
      case { 'bip', 'bipolar' };
        y = -1.0 + 2.0 ./ ( 1 + exp( -x ) );
      case { 'relu' }
        y = max( 0, x );
    end
end
