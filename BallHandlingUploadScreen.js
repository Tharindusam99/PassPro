import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    ActivityIndicator,
    Alert,
    Button,
    ScrollView,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Camera } from 'expo-camera';
import * as AV from 'expo-av';
import Ionicons from 'react-native-vector-icons/Ionicons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { BASEURL, getToken } from '../../../constants';

export default function BallHandlingUploadScreen({ navigation }) {
    const [correctVideo, setCorrectVideo] = useState(null);
    const [wrongVideo, setWrongVideo] = useState(null);
    const [isWrongVideoEnabled, setIsWrongVideoEnabled] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [hasCameraPermission, setHasCameraPermission] = useState(null);

    useEffect(() => {
        (async () => {
            const { status } = await Camera.requestCameraPermissionsAsync();
            setHasCameraPermission(status === 'granted');
        })();
    }, []);

    const pickVideo = async (type) => {
        let result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Videos,
            allowsEditing: true,
            aspect: [16, 9],
            quality: 1,
        });

        if (!result.canceled) {
            if (type === 'correct') {
                setCorrectVideo(result.assets[0].uri);
                setIsWrongVideoEnabled(true);
            } else {
                setWrongVideo(result.assets[0].uri);
            }
        }
    };

    const recordVideo = async (type) => {
        if (hasCameraPermission === null) {
            Alert.alert('Camera Permission', 'Requesting camera permission...');
            return;
        }
        if (hasCameraPermission === false) {
            Alert.alert('Camera Access Denied', 'Please allow camera access in settings.');
            return;
        }

        let result = await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Videos,
            quality: 1,
            allowsEditing: true,
        });

        if (!result.canceled) {
            if (type === 'correct') {
                setCorrectVideo(result.assets[0].uri);
                setIsWrongVideoEnabled(true);
            } else {
                setWrongVideo(result.assets[0].uri);
            }
        }
    };

    const uploadVideos = async () => {
        setIsLoading(true);
        try {
            if (!correctVideo || !wrongVideo) {
                Alert.alert('Error', 'Please select or record both videos.');
                setIsLoading(false);
                return;
            }

            const formData = new FormData();
            formData.append('ball_handling_correct', {
                uri: correctVideo,
                type: 'video/mp4',
                name: 'ball_handling_correct.mp4',
            });

            formData.append('ball_handling_wrong', {
                uri: wrongVideo,
                type: 'video/mp4',
                name: 'ball_handling_wrong.mp4',
            });

            const token = await getToken();
            if (!token) {
                console.error('No token found');
                setIsLoading(false);
                return;
            }

            const response = await fetch(`${BASEURL}ballhandling/upload`, {
                method: 'POST',
                body: formData,
                headers: {
                    'Authorization': `Bearer ${token}`,
                },
            });

            const data = await response.json();

            if (response.ok) {
                await AsyncStorage.setItem('BallHandlingId', data.data._id);
                Alert.alert('Success', 'Video uploaded successfully!', [
                    {
                        text: 'OK',
                        onPress: () =>
                            navigation.navigate('BallHandlingAnalyzeScreen', {
                                BallHandlingAnalyId: data.data._id,
                            }),
                    },
                ]);
            } else {
                Alert.alert('Error', `Upload failed: ${data.message}`);
            }
        } catch (error) {
            console.error('Upload error:', error);
            Alert.alert('Error', 'Failed to upload videos. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity
                    style={styles.backButton}
                    onPress={() => navigation.goBack()}
                >
                    <Ionicons name="arrow-back" size={24} color="#fff" />
                </TouchableOpacity>
                <Text style={styles.headerText}>Ball Handling Upload</Text>
            </View>

            <ScrollView contentContainerStyle={styles.scrollContainer} keyboardShouldPersistTaps="handled">
                {/* 📌 Pick or Record Ideal Video */}
                <Text style={styles.videoLabel}>Select Ideal Video:</Text>
                <View style={styles.buttonRow}>
                    <TouchableOpacity style={styles.fixedButton} onPress={() => pickVideo('correct')}>
                        <Text style={styles.buttonText}>Pick Ideal Video</Text>
                    </TouchableOpacity>
                    <View style={styles.buttonSpacer} />
                    <TouchableOpacity style={styles.fixedButton} onPress={() => recordVideo('correct')}>
                        <Text style={styles.buttonText}>Record Ideal Video</Text>
                    </TouchableOpacity>
                </View>

                {correctVideo && <AV.Video source={{ uri: correctVideo }} style={styles.video} useNativeControls resizeMode="contain" isLooping />}

                {/* 📌 Pick or Record Wrong Video */}
                <Text style={styles.videoLabel}>Select Your Video:</Text>
                <View style={styles.buttonRow}>
                    <TouchableOpacity
                        style={[styles.fixedButton, !isWrongVideoEnabled && styles.disabledButton]}
                        onPress={() => pickVideo('wrong')}
                        disabled={!isWrongVideoEnabled}
                    >
                        <Text style={styles.buttonText}>Pick Your Video</Text>
                    </TouchableOpacity>
                    <View style={styles.buttonSpacer} />
                    <TouchableOpacity
                        style={[styles.fixedButton, !isWrongVideoEnabled && styles.disabledButton]}
                        onPress={() => recordVideo('wrong')}
                        disabled={!isWrongVideoEnabled}
                    >
                        <Text style={styles.buttonText}>Record Your Video</Text>
                    </TouchableOpacity>
                </View>

                {wrongVideo && <AV.Video source={{ uri: wrongVideo }} style={styles.video} useNativeControls resizeMode="contain" isLooping />}

                {/* 📌 Upload Button */}
                <TouchableOpacity style={styles.uploadBtn} onPress={uploadVideos} disabled={isLoading}>
                    <Text style={styles.buttonText}>{isLoading ? 'Uploading...' : 'Upload Videos'}</Text>
                </TouchableOpacity>

                {isLoading && (
                    <View style={styles.loaderContainer}>
                        <ActivityIndicator size="large" color="#ef5350" />
                    </View>
                )}
            </ScrollView>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
    },
    scrollContainer: {
        flexGrow: 1,
        alignItems: 'center',
        paddingBottom: 60,
    },
    header: {
        width: '100%',
        backgroundColor: '#ef5350',
        padding: 20,
        flexDirection: 'row',
        alignItems: 'center',
    },
    backButton: {
        marginRight: 10,
        marginTop:20
    },
    headerText: {
        fontSize: 18,
        color: '#fff',
        fontWeight: '600',
        marginTop:20
    },
    videoLabel: {
        fontSize: 16,
        fontWeight: 'bold',
        marginTop: 10,
    },
    buttonRow: {
        flexDirection: 'row',
        justifyContent: 'center',
        width: '100%',
        marginVertical: 10,
    },
    fixedButton: {
        width: 160,
        paddingVertical: 15,
        backgroundColor: '#39c668',
        borderRadius: 5,
        alignItems: 'center',
    },
    disabledButton: {
        backgroundColor: '#ccc',
    },
    buttonSpacer: {
        width: 15,
    },
    uploadBtn: {
        marginTop: 20,
        backgroundColor: '#ef5350',
        padding: 15,
        borderRadius: 5,
        alignItems: 'center',
        width: '80%',
    },
    buttonText: {
        color: '#fff',
        fontSize: 12,
        fontWeight: 'bold',
    },
    video: {
        width: '90%',
        height: 200,
        backgroundColor: 'black',
    },
});
