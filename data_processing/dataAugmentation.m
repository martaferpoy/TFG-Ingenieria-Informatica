
% Cargar los coeficientes del filtro
load('num.mat');

% Directorio
myDir = "D:\TFG\filt_data_augmentation\train_data\";

myFiles = dir(fullfile(myDir,'*.mat'));

for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    fullFileName = fullfile(myDir, baseFileName);

    % Cargar datos
    load(fullFileName);
    
    % Eliminamos todas las señales que no tengan frecuencia 10
    if(fs_thorres == 10)
    
        % Aplicar coeficientes a la signal
        filtered_thorres = filtfilt(Num, 1, thorres);
        filtered_abdores = filtfilt(Num, 1, abdores);
        
        signal_size = size(filtered_abdores);
        % numero de instancias a eliminar (principio+final)
        remove_sides = 2000;
    
        % Numero de subsignals en las que dividir la signal (20 minutos
        % cada una)
        max_subsignals = floor((signal_size(2)-remove_sides)/fs_thorres/60/20);
        
        % Dimensiones de cada subsignal
        size_to_split = fs_thorres*60*20;
    
        % Dimensiones maximas de la signal completa para ser dividida en
        % subsignals de las mismas dimensiones
        max_size_signal = max_subsignals*size_to_split;
        
        % Instancias a eliminar al principio y al final
        remove_inst = (signal_size(2)-max_size_signal)/2;

        if(remove_inst <= 0)
            max_subsignals = max_subsignals - 1;
            max_size_signal = max_subsignals*size_to_split;
            remove_inst = (signal_size(2)-(max_size_signal))/2;
        end
        
        
        % Eliminar primeras y ultimas instancias (remove_inst)
        shortened_signal_thorres = filtered_thorres(remove_inst:(signal_size(2))-remove_inst-1);
        shortened_signal_abdores = filtered_abdores(remove_inst:(signal_size(2))-remove_inst-1);
        shortened_central_apneas = central_apneas(remove_inst:(signal_size(2))-remove_inst-1);
        shortened_hypopneas = hypopneas(remove_inst:(signal_size(2))-remove_inst-1);
        shortened_osa = obstructive_apneas(remove_inst:(signal_size(2))-remove_inst-1);
        
        % Dividir signal en subsignals (20 minutos) y almacenar en una matriz 
        [signal_matrix_thorres, signal_matrix_abdores, split_central_apnea, split_hypopneas, split_osa] = split_signals_20_mins(size_to_split, max_subsignals, shortened_signal_thorres, shortened_signal_abdores, shortened_central_apneas, shortened_hypopneas, shortened_osa);

        n_segments = max_subsignals;
        length_segments = size_to_split;

        % Estandarizar signal
        [stand_matrix_thorres,stand_matrix_abdores] = standardize_signal(signal_matrix_thorres, signal_matrix_abdores, n_segments);

        % Crear etiquetas para osa, hipopnea, central y todos los eventos
        vec_all_central = [];
        vec_all_hypo = [];
        vec_all_osa = [];
        vec_all_events = [];
        [vec_all_central, vec_all_hypo, vec_all_osa, vec_all_events] = label_signals(n_segments,size_to_split, vec_all_central, vec_all_hypo, vec_all_osa, vec_all_events, split_central_apnea, split_hypopneas, split_osa, ...
            fs_thorres);

        AHI = get_AHI(vec_all_events);


        if(AHI < 5)
            % Data augmentation
            % Eliminar primeras y ultimas instancias (remove_inst) + 6000
            % primeros (10 mins) + 6000 al final (10 mins)
            shortened_signal_thorres_augm = filtered_thorres(remove_inst + 6000:(signal_size(2))-remove_inst-1-6000);
            shortened_signal_abdores_augm = filtered_abdores(remove_inst + 6000:(signal_size(2))-remove_inst-1-6000);
            shortened_central_apneas_augm = central_apneas(remove_inst + 6000:(signal_size(2))-remove_inst-1-6000);
            shortened_hypopneas_augm = hypopneas(remove_inst + 6000:(signal_size(2))-remove_inst-1-6000);
            shortened_osa_augm = obstructive_apneas(remove_inst + 6000:(signal_size(2))-remove_inst-1-6000);
            

            % Dividir signal en subsignals (20 minutos) y almacenar en una matriz   
            [signal_matrix_thorres_augm, signal_matrix_abdores_augm, split_central_apnea_augm, split_hypopneas_augm, split_osa_augm] = split_signals_20_mins(size_to_split, ...
                max_subsignals-1,  shortened_signal_thorres_augm, shortened_signal_abdores_augm, shortened_central_apneas_augm, ...
                shortened_hypopneas_augm, shortened_osa_augm);
    
            n_segments_augm = max_subsignals - 1;
            length_segments = size_to_split;
    
            % Estandarizar signal
            [stand_matrix_thorres_augm,stand_matrix_abdores_augm] = standardize_signal(signal_matrix_thorres_augm, signal_matrix_abdores_augm, n_segments_augm);
    
           
            % Crear etiquetas para osa, hipopnea, central y todos los eventos
            [vec_all_central, vec_all_hypo, vec_all_osa, vec_all_events] = label_signals(n_segments_augm,size_to_split, vec_all_central, ...
                vec_all_hypo, vec_all_osa, vec_all_events, split_central_apnea_augm, split_hypopneas_augm, split_osa_augm, fs_thorres);


            stand_matrix_abdores = [stand_matrix_abdores stand_matrix_abdores_augm];
            stand_matrix_thorres = [stand_matrix_thorres stand_matrix_thorres_augm];

        end

        % Crear nombres de ficheros
        filename = ['D:\TFG\filt_data_augmentation\filt_train\filt_d_augm_' baseFileName];

        save(filename, "stand_matrix_thorres", "stand_matrix_abdores", "n_segments", "length_segments", "fs_thorres", "fs_abdores", "vec_all_central", "vec_all_hypo", "vec_all_osa", "vec_all_events")
    end



end

function [signal_matrix_thorres, signal_matrix_abdores, split_central_apnea, split_hypopneas, split_osa] = split_signals_20_mins(size_to_split, ...
    max_subsignals, shortened_signal_thorres, shortened_signal_abdores, shortened_central_apneas, shortened_hypopneas, shortened_osa)
    ncol = size_to_split;
    nrow = max_subsignals;

    
    signal_matrix_thorres = reshape(shortened_signal_thorres, [ncol, nrow]);
    signal_matrix_abdores = reshape(shortened_signal_abdores, [ncol, nrow]);
    
    split_central_apnea = reshape(shortened_central_apneas, [ncol, nrow]);
    split_hypopneas = reshape(shortened_hypopneas, [ncol, nrow]);
    split_osa = reshape(shortened_osa, [ncol, nrow]);
end

function [stand_matrix_thorres,stand_matrix_abdores] = standardize_signal(signal_matrix_thorres, signal_matrix_abdores, n_segments)

    %Estandarizar signal
    for segm = 1:n_segments
        if(segm == 1)
            stand_matrix_thorres = zscore(signal_matrix_thorres(:, segm)); 
            stand_matrix_abdores = zscore(signal_matrix_abdores(:, segm)); 
        else
            standarized_segment_t =  zscore(signal_matrix_thorres(:, segm));
            standarized_segment_a =  zscore(signal_matrix_abdores(:, segm));
            aa = size(stand_matrix_thorres);
            stand_matrix_thorres = [stand_matrix_thorres(:,1:aa(2)) standarized_segment_t];
            stand_matrix_abdores = [stand_matrix_abdores(:,1:aa(2)) standarized_segment_a];
        end
    end
end


function [vec_all_central, vec_all_hypo, vec_all_osa, vec_all_events] = label_signals(n_segments, size_to_split, vec_all_central, vec_all_hypo, vec_all_osa, vec_all_events, split_central_apnea, split_hypopneas, split_osa, ...
    fs_thorres)

% Recorremos los 26 bloques de 20 mins cada uno
for index = 1:n_segments

    % Vectores en los que vamos a almacenar las etiquetas y el numero de
    % veces q aparece cada una 
    vec_labels_central = [];
    vec_count_central = [];
    
    vec_labels_hypo = [];
    vec_count_hypo = [];
    
    vec_labels_osa = [];
    vec_count_osa = [];

    % primer elemento de cada segmento de 20 mins 
    % (lo vamos a usar para comparar
    central1 = split_central_apnea(1, index); 
    hypo1 = split_hypopneas(1, index); 
    osa1 = split_osa(1, index);

    % elemento previo al que se itera
    prev_central = central1;
    prev_hypo = hypo1;
    prev_osa = osa1;

    % conteo de instancias en las que hay evento
    count_osa = 1;
    count_hypopnea = 1;
    count_central = 1;

    % conteo de eventos
    count_osa_event = 0;
    count_hypopnea_event = 0;
    count_central_event = 0;

    % recorremos cada instante dentro de cada segmento de 20 mins
    for sec = 2:size_to_split
        % Conteo de numero de etiquetas 0 y 1 (no hay apnea, si hay apnea)
        if(split_central_apnea(sec, index) == prev_central)
            count_central = count_central + 1;
            if(sec == size_to_split)              
                vec_labels_central(length(vec_labels_central) + 1) = prev_central;
                vec_count_central(length(vec_count_central) + 1) = count_central;
            end
        else
            vec_labels_central(length(vec_labels_central) + 1) = prev_central;
            vec_count_central(length(vec_count_central) + 1) = count_central;
            prev_central = split_central_apnea(sec, index);
            count_central = 1;
            
        end

        if(split_hypopneas(sec, index) == prev_hypo)
            count_hypopnea = count_hypopnea + 1;
            if(sec == size_to_split)
                vec_labels_hypo(length(vec_labels_hypo) + 1) = prev_hypo;
                vec_count_hypo(length(vec_count_hypo) + 1) = count_hypopnea;
            end

        else
            vec_labels_hypo(length(vec_labels_hypo) + 1) = prev_hypo;
            vec_count_hypo(length(vec_count_hypo) + 1) = count_hypopnea;
            prev_hypo = split_hypopneas(sec, index);
            count_hypopnea = 1;
        end

        if(split_osa(sec, index) == prev_osa)
            count_osa = count_osa + 1;
            if(sec == size_to_split)
                vec_labels_osa(length(vec_labels_osa) + 1) = prev_osa;
                vec_count_osa(length(vec_count_osa) + 1) = count_osa;
            end    
            
        else
            vec_labels_osa(length(vec_labels_osa) + 1) = prev_osa;
            vec_count_osa(length(vec_count_osa) + 1) = count_osa;
            prev_osa = split_osa(sec, index);
            count_osa = 1;
        end

    end

    % Contar numero de unos con mas de 100 conteos

    for j = 2:(length(vec_labels_central)-1)
        if(vec_count_central(j) >= 10*fs_thorres && vec_labels_central(j) == 1)
            count_central_event = count_central_event + 1;
        end
    end

    for jj = 2:(length(vec_labels_hypo)-1)
        if(vec_count_hypo(jj) >= 10*fs_thorres && vec_labels_hypo(jj) == 1)
            count_hypopnea_event = count_hypopnea_event + 1;
        end
    end

    for jjj = 2:(length(vec_labels_osa)-1)
        if(vec_count_osa(jjj) >= 10*fs_thorres && vec_labels_osa(jjj) == 1)
            count_osa_event = count_osa_event + 1;
        end
    end


    % Aparte contamos los unos que estan al principio de la
    % señal, por si hay algun evento que se corta al recortar la señal

    if(vec_labels_central == 1)
        if(vec_count_central(1) <= 10*fs_thorres)
            count_central_event = count_central_event + (vec_count_central(1)/(10*fs_thorres));
        else
            count_central_event = count_central_event + 1;
        end
    end

    if(vec_labels_hypo == 1)
        if(vec_count_hypo(1) <= 10*fs_thorres)
            count_hypopnea_event = count_hypopnea_event + (vec_count_hypo(1)/(10*fs_thorres));
        else
            count_hypopnea_event = count_hypopnea_event + 1;
        end
    end

    if(vec_labels_osa == 1)
        if(vec_count_osa(1) <= 10*fs_thorres)
            count_osa_event = count_osa_event + (vec_count_osa(1)/(10*fs_thorres));
        else
            count_osa_event = count_osa_event + 1;
        end
    end

    % Hacemos lo mismo para la parte final
    if(~isempty(vec_labels_central))
        if(vec_labels_central(length(vec_labels_central)) == 1)
            if(vec_count_central(length(vec_labels_central)) <= 10*fs_thorres)
                count_central_event = count_central_event + (vec_count_central(length(vec_count_central))/(10*fs_thorres));

            else
                count_central_event = count_central_event + 1;
            end
        end
    end
    
    if(~isempty(vec_labels_hypo))
        if(vec_labels_hypo(length(vec_labels_hypo)) == 1)
            if(vec_count_hypo(length(vec_labels_hypo)) <= (10*fs_thorres))
                count_hypopnea_event = count_hypopnea_event + (vec_count_hypo(length(vec_count_hypo))/(10*fs_thorres));

            else
                count_hypopnea_event = count_hypopnea_event + 1;
            end
        end
    end

    if(~isempty(vec_labels_osa))
        if(vec_labels_osa(length(vec_labels_osa)) == 1)
            if(vec_count_osa(length(vec_labels_osa)) <= (10*fs_thorres))
                count_osa_event = count_osa_event + (vec_count_osa(length(vec_count_osa))/(10*fs_thorres));

            else
                count_osa_event = count_osa_event + 1;
            end
        end
    end

    
    vec_all_central(length(vec_all_central) + 1) = count_central_event;
    vec_all_hypo(length(vec_all_hypo) + 1) = count_hypopnea_event;
    vec_all_osa(length(vec_all_osa) + 1) = count_osa_event;

    vec_all_events(length(vec_all_events) + 1) = count_central_event + count_hypopnea_event + count_osa_event;

end

end


function AHI = get_AHI(vec_events)
    size_events = size(vec_events);
    AHI = sum(vec_events)/size_events(2)*3;
end