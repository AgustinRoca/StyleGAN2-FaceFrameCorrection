<template>
<v-form v-model="valid" @submit.prevent="submit">
    <v-container style="width:70vw">
        <v-row class="mb-5">
            <v-col>
                <v-text-field
                    v-model="id"
                    :rules="amountRules"
                    label="Face ID"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
        </v-row>
        <v-row class="mb-2">
            <h2>General features</h2>
        </v-row>
        <v-row class="mb-5">
            <v-col>
                <v-text-field
                    :value="ageAmount"
                    v-model="ageAmount"
                    label="Age"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="genderAmount"
                    v-model="genderAmount"
                    label="Gender"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
        </v-row>
        <v-row class="mb-2">
            <h2>Face Orientation</h2>
        </v-row>
        <v-row class="mb-5">
            <v-col>
                <v-text-field
                    :value="pitchAmount"
                    v-model="pitchAmount"
                    label="Vertical"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="yawAmount"
                    v-model="yawAmount"
                    label="Horizontal"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
        </v-row>
        <v-row class="mb-2">
            <h2>Eyes</h2>
        </v-row>
        <v-row class="mb-5">
            <v-col>
                <v-text-field
                    :value="eyeDistanceAmount"
                    v-model="eyeDistanceAmount"
                    label="Eye distance"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="eyeEyebrowDistanceAmount"
                    v-model="eyeEyebrowDistanceAmount"
                    label="Eye-eyebrow distance"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="eyeRatioAmount"
                    v-model="eyeRatioAmount"
                    label="Eye ratio"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="eyesOpenAmount"
                    v-model="eyesOpenAmount"
                    label="Eyes open"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="rollAmount"
                    v-model="rollAmount"
                    label="Roll"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
        </v-row>
        <v-row class="mb-2">
            <h2>Mouth</h2>
        </v-row>
        <v-row class="mb-5">
            <v-col>
                <v-text-field
                    :value="lipRatioAmount"
                    v-model="lipRatioAmount"
                    label="Lip ratio"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="mouthOpenAmount"
                    v-model="mouthOpenAmount"
                    label="Mouth open"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="mouthRatioAmount"
                    v-model="mouthRatioAmount"
                    label="Mouth ratio"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="smileAmount"
                    v-model="smileAmount"
                    label="Smile"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
        </v-row>
        <v-row class="mb-2">
            <h2>Nose</h2>
        </v-row>
        <v-row class="mb-5">
            <v-col>
                <v-text-field
                    :value="noseMouthDistanceAmount"
                    v-model="noseMouthDistanceAmount"
                    label="Nose-mouth distance"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="noseRatioAmount"
                    v-model="noseRatioAmount"
                    label="Nose ratio"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
            <v-col>
                <v-text-field
                    :value="noseTipAmount"
                    v-model="noseTipAmount"
                    label="Nose tip"
                    required
                    type="number"
                    density="compact"
                    hide-details="auto"
                ></v-text-field>
            </v-col>
        </v-row>

        <v-row>
            <v-col>
                <v-btn
                color="primary"
                class="mr-4"
                type="submit"
                :disabled="isLoading"
                >
                Generate
                </v-btn>
                <v-progress-circular
                    indeterminate
                    color="primary"
                    v-if="isLoading"
                ></v-progress-circular>
            </v-col>
        </v-row>
    </v-container>
  </v-form>
</template>

<script>
    import axios from 'axios'
    export default {
        data: () => ({
        refCount: 0,
        isLoading: false,
        valid: false,
        ageAmount: 0,
        eyeDistanceAmount: 0,
        eyeEyebrowDistanceAmount: 0,
        eyeRatioAmount: 0,
        eyesOpenAmount: 0,
        genderAmount: 0,
        lipRatioAmount: 0,
        mouthOpenAmount: 0,
        mouthRatioAmount: 0,
        noseMouthDistanceAmount: 0,
        noseRatioAmount: 0,
        noseTipAmount: 0,
        pitchAmount: 0,
        rollAmount: 0,
        smileAmount: 0,
        yawAmount: 0,
        id: '',
        amountRules: [
            v => !!v || 'Amount is required',
            v => Number.isInteger(v) > 0 || 'Amount must be an integer',
            v => Number(v) > 0 || 'Amount must be a positive integer',
        ],
        }),
        methods : {
            submit(){
                let formData = new FormData();
                formData.append('id', this.id)
                formData.append('ageAmount', this.ageAmount)
                formData.append('eyeDistanceAmount', this.eyeDistanceAmount)
                formData.append('eyeEyebrowDistanceAmount', this.eyeEyebrowDistanceAmount)
                formData.append('eyeRatioAmount', this.eyeRatioAmount)
                formData.append('eyesOpenAmount', this.eyesOpenAmount)
                formData.append('genderAmount', this.genderAmount)
                formData.append('lipRatioAmount', this.lipRatioAmount)
                formData.append('mouthOpenAmount', this.mouthOpenAmount)
                formData.append('mouthRatioAmount', this.mouthRatioAmount)
                formData.append('noseMouthDistanceAmount', this.noseMouthDistanceAmount)
                formData.append('noseRatioAmount', this.noseRatioAmount)
                formData.append('noseTipAmount', this.noseTipAmount)
                formData.append('pitchAmount', this.pitchAmount)
                formData.append('rollAmount', this.rollAmount)
                formData.append('smileAmount', this.smileAmount)
                formData.append('yawAmount', this.yawAmount)

                axios({
                    url: 'https://pf-2021-generadorcaras.it.itba.edu.ar/api/features',
                    method: 'post',
                    data: formData,
                    headers: { "Content-Type": "multipart/form-data" },
                }).then((response) => {
                    this.$emit('generated', response.data.img_bytes, response.data.z)
                }).catch((e) => {
                    alert('Error')
                    console.log(e)
                })
            },
            setLoading(isLoading) {
                if (isLoading) {
                    this.refCount++;
                    this.isLoading = true;
                } else if (this.refCount > 0) {
                    this.refCount--;
                    this.isLoading = (this.refCount > 0);
                }
            }
        },
        created() {
            axios.interceptors.request.use((config) => {
                this.setLoading(true);
                return config;
            }, (error) => {
                this.setLoading(false);
                return Promise.reject(error);
            });

            axios.interceptors.response.use((response) => {
                this.setLoading(false);
                return response;
            }, (error) => {
                this.setLoading(false);
                return Promise.reject(error);
            });
        }
    }
</script>