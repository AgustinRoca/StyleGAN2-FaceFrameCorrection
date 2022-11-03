<template>
<v-container style="width: 70vw">
    <v-form v-model="valid" @submit="submit">
            <v-row>
                <v-col md="9">
                    <v-file-input
                    counter
                    truncate-length="50"
                    v-model="image"
                    show-size
                    @change="preview_image"
                    @click:clear="preview_image"
                    accept="image/*"
                    />
                </v-col>
                <v-col md="2">
                    <v-btn
                    color="primary"
                    class="mr-4"
                    type="submit"
                    :disabled="image.length === 0 || isLoading"
                    >
                    Generate
                    </v-btn>
                </v-col>
                <v-col md="1">
                    <v-progress-circular
                                indeterminate
                                color="primary"
                                v-if="isLoading"
                    ></v-progress-circular>
                </v-col>
            </v-row>
            <v-row>
                <v-col md="9" class="d-flex justify-center">
                    <v-img contain max-width="50%" :src="url" />
                </v-col>
                <v-spacer></v-spacer>
            </v-row>
    </v-form>
</v-container>
</template>

<script>
    import axios from 'axios'
    export default {
        data: () => ({
            image: [],
            url: null,
            refCount: 0,
            isLoading: false,
            valid: false,
        }),
        methods : {
            submit(){
                let formData = new FormData();
                let reader = new FileReader();
                reader.readAsDataURL(this.image[0]);
                reader.onload = () => {
                    formData.append('file', reader.result)
                    axios({
                        url: 'https://pf-2021-generadorcaras.it.itba.edu.ar/api/latentspace',
                        method: 'post',
                        data: formData,
                        headers: { "Content-Type": "multipart/form-data" },
                    }).then((response) => {
                        this.$emit('generated', response.data.imgs_bytes, response.data.zs)
                    }).catch((e) => {
                        if(e.response == undefined){
                            alert('Error')
                        } else {
                            alert(e.response.data.message)
                        }
                    })
                };
                reader.onerror = function (error) {
                    console.log('Error: ', error);
                };
            },
            setLoading(isLoading) {
                if (isLoading) {
                    this.refCount++;
                    this.isLoading = true;
                } else if (this.refCount > 0) {
                    this.refCount--;
                    this.isLoading = (this.refCount > 0);
                }
            },
            preview_image() {
                if(this.image.length === 0) {
                    this.url = null
                    return;
                }
                this.url= URL.createObjectURL(this.image[0]);
            },
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
        },
    }
</script>