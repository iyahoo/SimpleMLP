classdef SimpleMLP
    properties ( Access = public )
        %% Data
        trainX  % Training data features matrix:
                % Row - each instance; Column - each feature
        trainY  % Training data class labels vector
        testX   %
        testY   %

        %% Initial parameters
        learning_rate
        epochs                % number of learning cycle
        count_hneuron         % Hidden neurons n or [n m ...]
        output_activation_fun %
        activation_fun        % 'sig', 'bip', or 'relu'. See `activationFunction.m`
        random_seed           % Integer or 'shuffle'

        %% Frequentry used variables ( calculated from Data and Initial parameters )
        labels                  % All class labels
        count_class
        count_neuron_per_layer  % Count of neurons in each layer
        count_layers

        %% Targets of train
        weights
        delta_weights
        neurons_output  % All ouputs of all neurons
        backprop_error  % backpropagated error

        %% Results
        RMS  % Root Mean Squere Error

        %% state
        is_trained
    end

    methods ( Access = public )
        %% Constructor
        function obj = SimpleMLP( varargin )
            defaults = struct( 'trainX', [], 'trainY', [], ...
                               'testX',  [], 'testY',  [], ...
                               'learning_rate', 0.01, ...
                               'count_hneuron', [5], ...
                               'epochs'       , 500, ...
                               'output_activation_fun', 'sig', ...
                               'activation_fun',  'sig', ...
                               'random_seed',   'shuffle' ...
                               );
            options = SimpleMLP.optionsMaker( defaults, varargin );

            obj = obj.initialize( options );
        end

        %% Main functions
        %% Train using `trainX` and `trainY`
        function new = train( obj )
            for c_epoch = 1 : obj.epochs % Current epoch
                if rem( c_epoch, 100 ) == 0
                    disp( [ 'Current epoch is ' int2str( c_epoch ) ] );
                end

                for c_inst = 1 : size( obj.trainX, 1 )
                    obj = obj.onePhaseTrain( c_inst );
                end

                %% Update RMS
                [ val RMS ] = obj.trainEvaluation();
                obj.RMS( c_epoch ) = RMS;
            end
            obj.is_trained = true;
            new = obj;
        end

        function output = predict( obj, input_features )
        %% Return raw outputs of output neurons (e.g. [-0.932 0.83 -0.39])
            assert( isnumeric( input_features ), ...
                    'input_features is type %s, not numeric', class( input_features ) );

            input_features_added = [ 1 input_features ]; % Add bias
            %% Copy cell `outputs_neuron` to local neurons for prediction
            neurons_output = obj.neurons_output;
            neurons_output{ 1 } = input_features_added;

            for c_layer = 2 : obj.count_layers
                neurons_output{ c_layer } = ...
                    neurons_output{ c_layer - 1 } * obj.weights{ c_layer - 1 };

                %% Not output layer
                if c_layer ~= obj.count_layers
                    neurons_output{ c_layer } = ...
                        SimpleMLP.activationFunction( neurons_output{ c_layer }, ...
                                                      obj.activation_fun );
                    %% Bais' output is 1
                    neurons_output{ c_layer }( 1 ) = 1;
                else
                    neurons_output{ c_layer } = ...
                        SimpleMLP.activationFunction( neurons_output{ c_layer }, ...
                                                      obj.output_activation_fun );
                end
            end

            output = neurons_output{ end };
        end
    end

    methods ( Access = protected )
        function obj = initialize( obj, options )
        %% Initialize parameters from argments
            assert( isstruct( options ), ...
                    'options is type %s, not struct', class( options ) );

            f = fieldnames( options );

            for i = 1 : length( f )
                %% Assign parameters to fields of model
                %% e.g. obj.trainX = options.trainX;
                eval( ['obj.' f{ i } ' = options.' f{ i } ';'] );
            end

            %% Assertion
            obj.assertion();

            %% Set random seed
            rng( obj.random_seed );

            %% Count of features
            count_input_neurons = size( obj.trainX, 2 );

            %% Calculate parameters
            obj.labels                 = unique( obj.trainY ); % `Unique` sort elements
            obj.count_class            = size( obj.labels, 1 );
            obj.count_neuron_per_layer = ...
                [ count_input_neurons, obj.count_hneuron, obj.count_class ];

            %% Construct and initialize network
            obj = obj ...
                  .addingBiasNeurons() ...
                  .convertLabels() ...
                  .initializeWeights() ...
                  .initializeNeuronOutputs();

            %% Bakcpropagation error cell size is same as `neurons_output`
            obj.backprop_error = obj.neurons_output;

            %% For record of each epoch
            obj.RMS = -1 * ones( 1, obj.epochs );

            %%
            obj.is_trained = false;
        end

        function obj = addingBiasNeurons( obj )
            obj.count_neuron_per_layer( 1 : end-1 ) = ...
                obj.count_neuron_per_layer( 1 : end - 1 ) + 1;
            obj.trainX       = [ ones( length( obj.trainX( :,1 ) ), 1 ) obj.trainX ];
            obj.testX        = [ ones( length( obj.testX( :,1 ) ), 1 )  obj.testX ];
            obj.count_layers = 2 + length( obj.count_hneuron );
        end

        function T = makeLabels( obj, target )
        % Label vector:
        % 1 2 3 1 2 3
        %
        % ||
        % \/
        %
        % Label matrix:
        %  1 -1 -1  1 -1 -1
        % -1  1 -1 -1  1 -1
        % -1 -1  1 -1 -1  1

            target_label = getfield( obj, target );
            count_data   = size( target_label, 1 );

            temp_T = zeros( obj.count_class, count_data );

            for i = 1 : count_data
                for j = 1 : obj.count_class
                    if obj.labels( j, 1 ) == target_label( i, 1 );
                        break;
                    end
                end
                temp_T( j, i ) = 1;
            end

            T = temp_T';
        end

        function obj = convertLabels( obj )
        % Converting labels in `trainY` and `testY`
            obj.trainY = obj.makeLabels( 'trainY' );
            obj.testY  = obj.makeLabels( 'testY' );
        end

        function obj = initializeWeights( obj )
            obj.weights       = cell( 1, obj.count_layers );
            obj.delta_weights = cell( 1, obj.count_layers );

            for i = 1 : length( obj.weights ) - 1
                %% Range of weights is [ -1,1]
                %% Weights between ith layer and i+1th layer
                obj.weights{ i } = ...
                    2 ...
                    * rand( obj.count_neuron_per_layer( i ), ...
                            obj.count_neuron_per_layer( i + 1 ) ) ...
                    - 1;
                %% Clear weights between ith layer bias and i+1th it
                obj.weights{ i }( :, 1 ) = 0;
                obj.delta_weights{ i }   = ...
                    zeros( obj.count_neuron_per_layer( i ), ...
                           obj.count_neuron_per_layer( i + 1 ) );

            end

            %% Output layer weights are counstant 1
            obj.weights{ end } = ones( obj.count_neuron_per_layer( end ), 1 );
        end

        function obj = initializeNeuronOutputs( obj )
            obj.neurons_output = cell( 1, obj.count_layers );

            for i = 1 : length( obj.neurons_output )
                obj.neurons_output{ i } = zeros( 1, obj.count_neuron_per_layer( i ) );
            end
        end


        %%%% Private methods For training
        function obj = onePhaseTrain( obj, c_inst )
            obj = obj.calculateNeuronsOutput( c_inst ) ...
                  .calculateBackpropagateError( c_inst ) ...
                  .calculateDeltaWeights() ...
                  .updateWeights() ...
                  .deltaWeightsReset();
        end

        function obj = calculateNeuronsOutput( obj, c_inst )
        %% Initialize 1st layer as a train instance
            obj.neurons_output{ 1 } = obj.trainX( c_inst, : );

            for c_layer = 2 : obj.count_layers
                obj.neurons_output{ c_layer } = ...
                    obj.neurons_output{ c_layer - 1 } * obj.weights{ c_layer - 1 };
                if c_layer ~= obj.count_layers
                    obj.neurons_output{ c_layer } = ...
                        SimpleMLP.activationFunction( obj.neurons_output{ c_layer }, ...
                                                      obj.activation_fun );
                    %% When no the last layer, output of biases converted by 1 ( 1 * 1 )
                    obj.neurons_output{ c_layer }( 1 ) = 1;
                else
                    obj.neurons_output{ c_layer } = ...
                        SimpleMLP.activationFunction( obj.neurons_output{ c_layer }, ...
                                                      obj.output_activation_fun );
                end
            end
        end

        function obj = calculateBackpropagateError( obj, c_inst )
        %% Final layer
            obj.backprop_error{ obj.count_layers } = ...
                obj.neurons_output{ obj.count_layers } - obj.trainY( c_inst, : );

            %% Others
            %% f'( u ) = gradient
            %% nodes_backpropagated_errors{ l } =
            %%     \sum_k{ \delta^{ l+1 }_{ k } * gradient_{ k } * w_{ ki } }
            for c_layer = obj.count_layers - 1 : -1 : 1
                gradient = ...
                    SimpleMLP.activationFunctionDrev( obj.neurons_output{ c_layer + 1 }, ...
                                                      obj.activation_fun );
                for neuron = 1 : length( obj.backprop_error{ c_layer } )
                    obj.backprop_error{ c_layer }( neuron ) = ...
                        sum( obj.backprop_error{ c_layer + 1 } ...
                             .* gradient ...
                             .* obj.weights{ c_layer }( neuron, : ) );
                end
            end
        end

        function obj = calculateDeltaWeights( obj )
            for c_layer = obj.count_layers : -1 : 2
                gradient = SimpleMLP.activationFunctionDrev( obj.neurons_output{ c_layer }, ...
                                                             obj.activation_fun );
                obj.delta_weights{ c_layer - 1 } = ...
                    obj.delta_weights{ c_layer - 1 } ...
                    + obj.neurons_output{ c_layer - 1 }' ...
                    * ( obj.backprop_error{ c_layer } ...
                        .* gradient );
            end

            for c_layer = 1 : obj.count_layers - 1
                obj.delta_weights{ c_layer } = ...
                    obj.learning_rate * obj.delta_weights{ c_layer };
            end
        end

        function obj = updateWeights( obj )
            for c_layer = 1 : obj.count_layers - 1
                obj.weights{ c_layer } = ...
                    obj.weights{ c_layer } - obj.delta_weights{ c_layer };
            end
        end

        function obj = deltaWeightsReset( obj )
            for c_layer = 1 : length( obj.delta_weights )
                obj.delta_weights{ c_layer } = 0 * obj.delta_weights{ c_layer };
            end
        end
    end

    % Helper
    methods ( Access = public, Static )
        % For making default variables
        % implemented in file optionsMaker.m
        [ options field_names ] = optionsMaker( defaults, vara );

        % Activation Function
        y = activationFunction( x, selector );
        y = activationFunctionDrev( x, selector );
    end

    methods ( Access = private )
        function assertion( obj )
        %% This is called when a new SimpleMLP class is instantiation
            assert( ~isempty( obj.trainX ), 'trainX is not defined.' );
            assert( ~isempty( obj.trainY ), 'trainY is not defined.' );
            assert( ~isempty( obj.testX ), 'testX is not defined.' );
            assert( ~isempty( obj.testY ), 'testY is not defined.' );

            assert( isequal( size( obj.trainX, 1 ), size( obj.trainY, 1 ) ), ...
                    ['The number of trainX and trainY instance have to be same.']);

            assert( isequal( size( obj.testX, 1 ), size( obj.testY, 1 ) ), ...
                    ['The number of testX and testY instance have to be same.']);

            obj.isaWithAssert( 'trainX', 'matrix' );
            obj.isaWithAssert( 'trainY', 'matrix' );
            obj.isaWithAssert( 'testX',  'matrix' );
            obj.isaWithAssert( 'testX',  'matrix' );

            %% Initial parameters
            obj.isaWithAssert( 'learning_rate', 'double' );
            obj.isaWithAssert( 'count_hneuron', 'double' );
            obj.isaWithAssert( 'epochs', 'double' );
            obj.isaWithAssert( 'activation_fun', 'char' );
        end

        function isaWithAssert( obj, property, is )
            subject = getfield( obj, property );

            switch is
              case { 'matrix' }
                p = ismatrix( subject );
              otherwise
                p = isa( subject, is );
            end

            assert( p, [ property 'is type %s, not ' is ], class( subject ));
        end
    end
end
