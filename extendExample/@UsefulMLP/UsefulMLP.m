classdef UsefulMLP < SimpleMLP
    methods ( Access = public )
        function obj = UsefulMLP( varargin )
            defaults = struct( 'trainX', [], 'trainY', [], ...
                               'testX',  [], 'testY',  [], ...
                               'learning_rate', 0.01, ...
                               'count_hneuron', [5], ...
                               'epochs'       , 500, ...
                               'output_activation_fun', 'sig', ...
                               'activation_fun',        'sig', ...
                               'random_seed', 12345 ...
                               );
            opt = UsefulMLP.optionsMaker( defaults, varargin );

            obj@SimpleMLP( 'trainX', opt.trainX, ...
                           'trainY', opt.trainY, ...
                           'testX',  opt.testX, ...
                           'testY',  opt.testY, ...
                           'learning_rate', opt.learning_rate, ...
                           'count_hneuron', opt.count_hneuron, ...
                           'epochs',        opt.epochs, ...
                           'output_activation_fun', opt.output_activation_fun, ...
                           'activation_fun',        opt.activation_fun, ...
                           'random_seed',           opt.random_seed ...
                           );
        end

        %% For calculating accuracy
        function [ acc, RMS ] = evaluation( obj, targetX, targetY )
        % acc: 1 - miss ratio
        % RMS: sqrt( ( sum_i^n( ( output_i - T_i ) ^ 2 ) * 0.5 ) * 2 )
            assert( isnumeric( targetX ), ...
                    'targetX is type %s, not numeric', class( targetX ) );
            assert( isnumeric( targetY ), ...
                    'targetX is type %s, not numeric', class( targetX ) )

            miss_count  = 0;
            RMS         = 0;
            target_size = size( targetX, 1 );
            
            for c_inst = 1 : target_size
                output = obj.predict( targetX( c_inst, : ) );
                T      = targetY( c_inst, : );

                [ val actual ]   = max( output );
                [ val expected ] = max( T );

                if actual ~= expected
                    miss_count = miss_count + 1;
                end

                RMS = RMS + sum( ( output - T ) .^ 2) * 0.5;
            end

            RMS = sqrt( RMS / ( 2 * target_size ) );
            acc = 1 - miss_count / target_size;
        end

        function [ acc, RMS ] = trainEvaluation( obj )
        %% drop bias from trainX because function obj.predict add bias
            [ acc, RMS ] = obj.evaluation( obj.trainX( :, 2 : end ), obj.trainY );
        end

        function [ acc, RMS ] = testEvaluation( obj )
            [ acc, RMS ] = obj.evaluation( obj.testX( :, 2 : end ), obj.testY );
        end

        function showRMS( obj, varargin )
        % Show graph using parameter RMS( 1 : end )
            defaults = struct( 'title_str', ...
                               [ 'RMS: ' class( obj ) ...
                                 ' L = ' int2str( obj.count_hneuron ) ...
                                 ' Epochs = ' int2str( obj.epochs ) ...
                                 ' LearningRate = ' num2str( obj.learning_rate ) ], ...
                               'xlabel_str', 'Number of Epochs', ...
                               'ylabel_str', 'RMS', ...
                               'axis_range', [ 0 obj.epochs 0 1 ], ...
                               'save_file_name', '', ...
                               'expansion', 'epsc' ...
                               );
            [ options f ] = SimpleMLP.optionsMaker( defaults, varargin );

            for i = 1 : length( f );
                eval( [ f{ i }, ' = options.', f{ i }, ';' ] );
            end

            figure;
            plot( 1 : obj.epochs, obj.RMS );
            hold off;
            title( title_str );
            xlabel( xlabel_str );
            ylabel( ylabel_str );
            axis( axis_range );

            if ~isempty( save_file_name )
                saveas( gcf, save_file_name, expansion )
                csvwrite( [ save_file_name '.csv' ], obj.RMS  )
            end
        end

        function saveModel( obj, save_file_name )
            assert( ischar( save_file_name ), ...
                    'save_file_name is type %s, not char', class( save_file_name ) );

            save( [ save_file_name '.mat' ] , 'obj' );
        end
    end
end
