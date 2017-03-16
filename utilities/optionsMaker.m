function [ options field_names ] = optionsMaker(defaults, vara)
%% Call:
%%   options = optionsMaker(defaults, vara)
%%   defaults: Structure of default values
%%   vara: Actual argments cell. `Varargin` is used usually.
%% Usage:
%%    This methods is proposed in
%%      https://stackoverflow.com/questions/2775263/how-to-deal-with-name-value-pairs-of-function-arguments-in-matlab
%%
%%    defaults = struct('delimiter', ',');
%%    options = optionsMaker(defaults, varargin);
%%    delim = getfield(options, 'delimiter');

    % Type Check
    assert( isstruct(defaults), 'default is type %s, not struct', class( defaults ) );
    assert( iscell(vara), 'default is type %s, not cell', class( vara ) );

    options = defaults;
    option_name = fieldnames(options);
    n_args = length(vara);
    if round(n_args / 2) ~= n_args / 2
        error('normalize needs property_name/propaerty_value pairs')
    end

    for pair = reshape(vara, 2, [])
        inp_name = pair{1};

        if any(strcmp(inp_name,option_name))
            options.(inp_name) = pair{2};
        else
            error('%s is not a recognized parameter name', inp_name);
        end
    end

    field_names = fieldnames( options );
end
